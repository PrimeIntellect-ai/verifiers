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


ConfigT = TypeVar("ConfigT", bound=UserConfig)


class User(ServerBase[ConfigT, StateT]):
    async def respond(self, message: str) -> Messages:
        """Return the next user messages; an empty message opens a task without a prompt."""
        raise NotImplementedError

    def _register(self, mcp: FastMCP) -> None:
        from verifiers.v1.dialects.chat import message_to_wire

        user = self
        last: tuple[int, str, str] | None = None
        lock = asyncio.Lock()

        async def respond(message: str, seq: int = -1) -> str:
            # Replay cache: the host retries a turn whose response was lost on the wire. The
            # simulator already advanced for that turn, so serve the recorded payload instead
            # of advancing it twice. `seq` is the caller's conversation position. The lock
            # serializes duplicate in-flight attempts (a retry racing a slow first execution)
            # so they join the recorded turn rather than each advancing the simulator.
            nonlocal last
            async with lock:
                if seq >= 0 and last is not None and last[:2] == (seq, message):
                    return last[2]
                messages = await user.respond(message)
                wire = [
                    m if isinstance(m, dict) else message_to_wire(m) for m in messages
                ]
                payload = json.dumps({"messages": wire})
                last = (seq, message, payload)
                return payload

        mcp.add_tool(self._with_state(respond), name="respond")
