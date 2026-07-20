"""Framework-driven user simulation.

When `TaskData.prompt` is `None`, `respond("")` supplies the opening message. A simulator ends
the interaction by setting shared state that the task checks with `@stop`.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, TypeVar

from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.state import StateT
from verifiers.v1.types import Messages

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


class UserConfig(BaseConfig):
    """Placement for a host-driven simulator.

    By default it runs in `runtime`; `colocated` reuses the harness runtime while remaining
    reachable from the host.
    """

    colocated: bool = False
    runtime: RuntimeConfig = SubprocessConfig()
    timeout: float = 60.0
    """Bound on one `respond()` attempt (connect + call). A wedged connection fails the attempt
    so the host's retry can recover it within the harness window; the turn's worst case is
    this times the retry budget. Raise it for simulators whose turns legitimately run long
    (e.g. model-backed users)."""


ConfigT = TypeVar("ConfigT", bound=UserConfig)


class User(ServerBase[ConfigT, StateT]):
    async def respond(self, message: str) -> Messages:
        """Return the next user messages; an empty message opens a task without a prompt."""
        raise NotImplementedError

    def _register(self, mcp: FastMCP) -> None:
        from verifiers.v1.dialects.chat import message_to_wire

        user = self
        last_turn: tuple[int, str] | None = None
        last_payload = ""
        lock = asyncio.Lock()

        async def advance(message: str) -> str:
            messages = await user.respond(message)
            wire = [m if isinstance(m, dict) else message_to_wire(m) for m in messages]
            return json.dumps({"messages": wire})

        # State sync (pull/commit) lives *inside* the lock, so `advance` is one atomic turn.
        synced = self._with_state(advance)

        async def respond(message: str, seq: int = -1) -> str:
            # Replay cache: the host retries a turn whose response was lost on the wire. The
            # simulator already advanced for that turn, so serve the recorded payload instead
            # of advancing it twice. `seq` is the caller's conversation position. The whole turn
            # — including the shared-state commit — runs under the lock, and the cache is
            # published only after it commits, so a racing retry (racing a slow first attempt)
            # either drives a fresh turn or joins the fully-committed one, never a mid-commit
            # read. The server process is per-rollout (`serve_user`), so cache and lock span
            # exactly one conversation.
            nonlocal last_turn, last_payload
            async with lock:
                if seq >= 0 and (seq, message) == last_turn:
                    return last_payload
                last_payload = await synced(message)
                last_turn = (seq, message)
                return last_payload

        mcp.add_tool(respond, name="respond")
