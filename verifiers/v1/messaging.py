"""The `message_agent` tool: an env episode's agent steers a sibling agent.

Enabled per agent by `AgentConfig.message_agents`. The rollout serves this toolset
next to its task's tools; a call posts `{seat, text}` to the interception's `/steer`
route with the rollout's private state bearer, and the interception steers that seat
of the same episode (`vf.steer`), recording the caller's seat as the sender.
"""

import json
import os

from verifiers.v1.mcp.server import _channel_request
from verifiers.v1.mcp.toolset import Toolset, ToolsetConfig
from verifiers.v1.utils.decorators import tool


class AgentMessaging(Toolset[ToolsetConfig]):
    TOOL_PREFIX = "agents"

    @tool
    async def message_agent(self, seat: str, text: str) -> str:
        """Send a message to another agent of this episode, by its seat name. It reaches
        that agent as a user message before its next model call."""
        state_url = os.environ.get("VF_STATE_URL", "")
        if not state_url.endswith("/state"):
            return "not delivered: no messaging channel"
        response = await _channel_request(
            "POST",
            state_url[: -len("/state")] + "/steer",
            os.environ.get("VF_STATE_SECRET", ""),
            content=json.dumps({"seat": seat, "text": text}).encode(),
            client=self._state_client,
        )
        receipt = response.json()
        if receipt.get("queued"):
            return f"delivered to {seat}"
        return f"not delivered: {receipt.get('reason')}"


if __name__ == "__main__":
    AgentMessaging.run()
