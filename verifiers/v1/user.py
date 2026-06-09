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

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from verifiers.v1.runtimes import Runtime
from verifiers.v1.tools import Tools, serve_tools
from verifiers.v1.types import Messages

logger = logging.getLogger(__name__)


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
    (`{"messages": [...], "done": bool}`) into typed `(messages, done)`."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from verifiers.v1.interception import parse_message

    async with contextlib.AsyncExitStack() as stack:
        read, write, *_ = await stack.enter_async_context(streamable_http_client(url))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        async def respond(message: str) -> tuple[Messages, bool]:
            result = await session.call_tool("respond", {"message": message})
            texts = [
                b.text for b in result.content if getattr(b, "type", None) == "text"
            ]
            data = json.loads("\n".join(texts))
            messages = [parse_message(m) for m in data["messages"]]
            return messages, bool(data["done"])

        yield respond


@contextlib.asynccontextmanager
async def serve_user(
    user: Tools | None, runtime: Runtime
) -> AsyncIterator[Respond | None]:
    """Bring a rollout's user server up (colocated in the harness's runtime, like a tool
    server) and yield the async `respond` for the interception server to drive — or `None`
    when the taskset has no user server. Colocated keeps it reachable from the host (where
    the interception server runs) for the subprocess/docker runtimes."""
    if user is None:
        yield None
        return
    async with serve_tools([user], runtime, colocated=True) as urls:
        async with connect_user(next(iter(urls.values()))) as respond:
            yield respond
