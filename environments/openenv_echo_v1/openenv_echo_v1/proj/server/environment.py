from typing import Any
from uuid import uuid4

from fastmcp import FastMCP
from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State


class EchoEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("echo_env")

        @mcp.tool
        def echo_message(message: str) -> str:
            """Return the message unchanged."""
            return message

        @mcp.tool
        def echo_with_length(message: str) -> dict:
            """Return the message and its character count."""
            return {"message": message, "length": len(message)}

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()))

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        self._state = State(episode_id=episode_id or str(uuid4()))
        return Observation(
            reward=0.0,
            metadata={"status": "ready", "seed": seed},
        )

    def _step_impl(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return Observation(
            metadata={"error": f"Unsupported action: {type(action).__name__}"}
        )

    @property
    def state(self) -> State:
        return self._state
