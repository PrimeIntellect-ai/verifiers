"""`User` + `UserConfig`: the user simulator authored as a vf-native class with a `respond` hook.

A taskset registers a `User` via `Taskset.user` — a vf-native class (like `Toolset`, served as an
MCP server with a runtime) exposing a single `respond` tool. Unlike a tool server it is never handed
to the model: the framework drives it. After each model turn the interception server calls `respond`
with the model's last message, appends the simulated user message(s), and re-prompts — so a
multi-turn game plays out as alternating assistant/user turns in the trace, the harness none the
wiser. When the task carries no prompt (`task.instruction is None`), the simulator also opens the
conversation: the interception server calls `respond("")` once before the first model turn and seeds
its reply as the initial user message. The host side that drives it lives in `launch` (`serve_user` /
`connect_user`).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypeVar

from pydantic_config import BaseConfig

from verifiers.v1.mcp.server import ServerBase
from verifiers.v1.runtimes import RuntimeConfig, SubprocessConfig
from verifiers.v1.types import Messages

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


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
        """The model's last assistant text in → the next user message(s) + a done flag out.
        Called once with an empty `message` to open the conversation when the task has no
        prompt (`task.instruction is None`)."""
        raise NotImplementedError

    def _register(self, mcp: FastMCP) -> None:
        from verifiers.v1.dialects.chat import message_to_wire

        user = self

        async def respond(message: str) -> str:
            messages, done = await user.respond(message)
            wire = [m if isinstance(m, dict) else message_to_wire(m) for m in messages]
            return json.dumps({"messages": wire, "done": done})

        mcp.add_tool(respond, name="respond")
