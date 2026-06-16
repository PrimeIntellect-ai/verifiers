"""The user simulator: a first-class conversation partner, served like a tool server.

A taskset registers a `User` via `Taskset.user` — a vf-native class (like `Toolset`,
served as an MCP server with a runtime) exposing a single `respond` tool. Unlike a tool server,
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

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from pydantic_config import BaseConfig

from verifiers.v1.errors import ProgramError, RolloutError
from verifiers.v1.runtimes import Runtime, RuntimeConfig, SubprocessConfig
from verifiers.v1.tools import ServerBase, serve
from verifiers.v1.types import Messages

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class UserConfig(BaseConfig):
    """Where the user simulator runs (placement). The framework always drives it from the host.
    Default — its own host (`subprocess`) runtime — runs it where `verifiers` + the env module
    already live, reachable from any harness runtime (nothing to fetch). Set `colocated` to run it
    inside the harness's runtime instead (in a sandbox that uploads + installs the env package, a
    per-rollout cost). Subclass to add the user's own knobs (the data its `respond` reads)."""

    colocated: bool = False
    """Run the user simulator inside the harness's runtime, reusing it (its port is published
    back to the host so the framework can still drive it). Off by default — see `ToolsetConfig`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The user simulator's own runtime, used unless `colocated` (host/subprocess by default)."""


ConfigT = TypeVar("ConfigT", bound=UserConfig)

# The colocated user server is up once its in-runtime probe passes, but under high concurrency
# it can still momentarily refuse a host connection. Retry the connect before giving up so a
# transient refusal doesn't fail the rollout.
_USER_CONNECT_ATTEMPTS = 12
_USER_CONNECT_BACKOFF = 0.2  # seconds, exponential up to the cap
_USER_CONNECT_MAX_BACKOFF = 2.0


class User(ServerBase[ConfigT]):
    """A user simulator authored as a vf-native class, initialized from its config: implement
    `respond` (the model's last message in → the next user message(s) + a done flag out).
    Consumed by the framework (the interception server drives it), never shown to the model.
    Example:

        class HagglerUserConfig(vf.UserConfig):
            target_price: int = 0

        class HagglerUser(vf.User[HagglerUserConfig]):
            async def respond(self, message: str) -> tuple[vf.Messages, bool]:
                ...
                return [{"role": "user", "content": reply}], done
    """

    async def respond(self, message: str) -> tuple[Messages, bool]:
        raise NotImplementedError

    def _register(self, mcp: FastMCP) -> None:
        from verifiers.v1.dialects.chat import message_to_wire

        user = self

        async def respond(message: str) -> str:
            messages, done = await user.respond(message)
            wire = [m if isinstance(m, dict) else message_to_wire(m) for m in messages]
            return json.dumps({"messages": wire, "done": done})

        mcp.add_tool(respond, name="respond")


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

    from verifiers.v1.dialects import parse_message

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
    user: User | None,
    task,
    agent_runtime: Runtime | None = None,
) -> AsyncIterator[Respond | None]:
    """Bring a rollout's user server up (via the shared `serve` launcher, `for_host=True` since
    the framework drives the user from the HOST) and yield the async `respond` the interception
    server drives — or `None` when the taskset has no user server. Placement is the user's
    `config` (colocated in the agent's runtime, or its own); the rollout's `task` is shipped to
    the server for its `setup`."""
    if user is None:
        yield None
        return
    async with serve(user, task, agent_runtime, for_host=True) as url:
        async with connect_user(url) as respond:
            yield respond
