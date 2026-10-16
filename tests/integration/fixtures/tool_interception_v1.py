"""Native and MCP tool gates preserve allow/deny/stop without executing denied calls."""

from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

MARKER = "vf-tool-gate-executed"


class GateToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "gate"

    @vf.tool
    def touch(self) -> str:
        """Perform the test action and return its private execution stamp."""
        Path(MARKER).write_text("executed")
        return "executed-gate-7f3"


class GateData(vf.TaskData):
    action: Literal["allow", "deny", "stop"]


class GateTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(colocated=True)


class GateTask(vf.Task[GateData, vf.State, GateTaskConfig]):
    @classmethod
    def toolsets(cls, config: GateTaskConfig) -> list[vf.Toolset]:
        return [GateToolset(config.tools)]

    @vf.stop
    def stop_tool(self, request: vf.Request) -> bool:
        return bool(
            self.data.action == "stop"
            and request.messages
            and isinstance(request.messages[-1], vf.ToolMessage)
            and not request.messages[-1].content
        )

    @vf.intercept
    def deny_tool(self, request: vf.Request) -> vf.Request | None:
        if self.data.action != "deny" or not request.messages:
            return None
        result = request.messages[-1]
        if isinstance(result, vf.ToolMessage) and not result.content:
            return request.model_copy(
                update={
                    "messages": [
                        *request.messages[:-1],
                        result.model_copy(update={"content": "GATE_BLOCKED"}),
                    ]
                }
            )

    @vf.reward
    async def gated(self, runtime: vf.Runtime) -> float:
        executed = (await runtime.run(["test", "-e", MARKER], {})).exit_code == 0
        return float(executed == (self.data.action == "allow"))


class GateConfig(vf.TasksetConfig):
    task: GateTaskConfig = GateTaskConfig()
    action: Literal["allow", "deny", "stop"] = "allow"
    target: Literal["native", "mcp"] = "native"


class ToolInterceptionTaskset(vf.Taskset[GateTask, GateConfig]):
    def load(self) -> list[GateTask]:
        action = (
            f"Use your native shell tool exactly once to run `touch {MARKER}`. Do not use MCP tools."
            if self.config.target == "native"
            else "Call the touch tool from the gate MCP server exactly once."
        )
        return [
            GateTask(
                GateData(
                    idx=0,
                    action=self.config.action,
                    prompt=(
                        f"{action} Whether the call succeeds or is blocked, do not retry or use another tool; reply DONE."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["ToolInterceptionTaskset"]


if __name__ == "__main__":
    GateToolset.run()
