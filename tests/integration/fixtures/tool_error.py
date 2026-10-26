"""tool-error (v1, MCP tool): a tool's exception text reaches the model.

Task tools raise to tell the model what went wrong (an unknown id, an invalid argument),
so the exception's message must come back as the tool result. The tool always raises
with a token the prompt never reveals, so the reward is 1.0 only if the error text
reached the trace.
"""

import verifiers.v1 as vf
from verifiers.v1.types import content_text

ERROR_TOKEN = "err-5c2"  # only the raised message carries this


class OrdersToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "orders"

    @vf.tool
    def lookup(self, order_id: str) -> str:
        """Look up an order by its id."""
        raise ValueError(f"Order {order_id} not found [{ERROR_TOKEN}]")


class ToolErrorTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class ToolErrorTask(vf.Task[vf.TaskData, vf.State, ToolErrorTaskConfig]):
    @classmethod
    def toolsets(cls, config: ToolErrorTaskConfig) -> list[vf.Toolset]:
        return [OrdersToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def error_reported(self, trace: vf.Trace) -> float:
        results = (content_text(m.content) for m in trace.tool_messages)
        return float(any(ERROR_TOKEN in r for r in results))


class ToolErrorConfig(vf.TasksetConfig):
    task: ToolErrorTaskConfig = ToolErrorTaskConfig()


class ToolErrorTaskset(vf.Taskset[ToolErrorTask, ToolErrorConfig]):
    def load(self) -> list[ToolErrorTask]:
        return [
            ToolErrorTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Call the `lookup` tool from the `orders` MCP server with the order "
                        'id "A-17", then reply with exactly what it returns inside '
                        "<answer></answer> tags."
                    ),
                ),
                self.config.task,
            )
        ]


__all__ = ["ToolErrorTaskset"]


if __name__ == "__main__":
    OrdersToolset.run()
