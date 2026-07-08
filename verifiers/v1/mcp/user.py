"""`User` + `UserConfig`: the user simulator authored as a vf-native class with a `respond` hook.

A task registers a `User` via `Task.user` — a vf-native class (like `Toolset`, served as an
MCP server with a runtime) exposing a single `respond` tool. Unlike a tool server it is never handed
to the model: the framework drives it. After each model turn the interception server calls `respond`
with the model's last message, appends the simulated user message(s), and re-prompts — so a
multi-turn game plays out as alternating assistant/user turns in the trace, the harness none the
wiser. When the task carries no prompt (`task.prompt is None`), the simulator also opens the
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
from verifiers.v1.state import StateT
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


class User(ServerBase[ConfigT, StateT]):
    """A user simulator authored as a vf-native class, initialized from its config: implement
    `respond` (the model's last message in → the next user message(s) out). Consumed by the framework
    (the interception server drives it), never shown to the model. To end the trajectory, set a flag
    on the shared `self.state` and have the task declare a `@vf.stop` over it (the framework holds
    no built-in end signal — see `verifiers.v1.state`). Example:

        class HagglerState(vf.State):
            deal_closed: bool = False

        class HagglerUser(vf.User[vf.UserConfig, HagglerState]):
            async def respond(self, message: str) -> vf.Messages:
                ...
                if deal_done:
                    self.state.deal_closed = True   # the task's @vf.stop ends it on this
                return [{"role": "user", "content": reply}]

        class HagglerTask(vf.Task[HagglerState]):
            def user(self) -> vf.User:
                return HagglerUser(self.user_config)

            @vf.stop
            async def deal_closed(self, trace) -> bool:
                return trace.state.deal_closed

    Parameterize the user with the same `State` subclass — `User[Config, MyState]` — so `self.state`
    is typed; it defaults to the base `State`.
    """

    async def respond(self, message: str) -> Messages:
        """The model's last assistant text in → the next user message(s) out. Called once with an
        empty `message` to open the conversation when the task has no prompt (`task.prompt is
        None`); end the trajectory by setting a `self.state` flag a task `@vf.stop` checks."""
        raise NotImplementedError

    def _register(self, mcp: FastMCP) -> None:
        from verifiers.v1.dialects.chat import message_to_wire

        user = self

        async def respond(message: str) -> str:
            messages = await user.respond(message)
            wire = [m if isinstance(m, dict) else message_to_wire(m) for m in messages]
            return json.dumps({"messages": wire})

        mcp.add_tool(self._with_state(respond), name="respond")
