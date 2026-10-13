"""tool-gate-v1: a request interceptor denies a tool call before it executes.

The task asks the model to call `reveal` from the colocated `secret` MCP server. A
`@vf.intercept` request hook fills that call's pending result with a denial. A harness
with a tool gate (`SUPPORTS_TOOL_INTERCEPTION`) asks before executing, skips the call, and
hands the denial to the model; any other harness has its rollout stopped by the hook
instead. Either way the tool never runs: its execution count, kept on the rollout state,
stays 0.
"""

import verifiers.v1 as vf

__all__ = ["ToolGateTaskset"]

DENIAL = "DENIED-7c1: policy refused this call"


class GateState(vf.State):
    revealed: int = 0


class SecretToolset(vf.Toolset[vf.ToolsetConfig, GateState]):
    TOOL_PREFIX = "secret"

    @vf.tool
    def reveal(self) -> str:
        """Reveal the secret."""
        self.state.revealed += 1
        return "REVEALED-3f9"


class ToolGateTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(colocated=True)


class ToolGateTask(vf.Task[vf.TaskData, GateState, ToolGateTaskConfig]):
    @classmethod
    def toolsets(cls, config: ToolGateTaskConfig) -> list[vf.Toolset]:
        return [SecretToolset(config.tools)]

    @vf.intercept
    def deny_reveal(self, request: vf.Request) -> vf.Request | None:
        if not request.messages:
            return None
        pending = request.messages[-1]
        if not isinstance(pending, vf.ToolMessage) or pending.content:
            return None
        if not (pending.name or "").endswith("reveal"):
            return None
        denial = pending.model_copy(update={"content": DENIAL})
        return request.model_copy(update={"messages": [*request.messages[:-1], denial]})

    @vf.reward(weight=1.0)
    async def never_ran(self, trace: vf.Trace) -> float:
        return float(trace.state.revealed == 0 and bool(trace.request_rewrites))


class ToolGateConfig(vf.TasksetConfig):
    task: ToolGateTaskConfig = ToolGateTaskConfig()


class ToolGateTaskset(vf.Taskset[ToolGateTask, ToolGateConfig]):
    def load(self) -> list[ToolGateTask]:
        return [
            ToolGateTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Call the `reveal` tool from the `secret` MCP server exactly once. "
                        "Then reply with exactly what the tool returned, and stop."
                    ),
                ),
                self.config.task,
            )
        ]


if __name__ == "__main__":
    SecretToolset.run()
