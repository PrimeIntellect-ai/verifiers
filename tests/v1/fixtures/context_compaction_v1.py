"""Context-compaction E2E scenarios for harness agent loops."""

from typing import Literal

from pydantic import Field

import verifiers.v1 as vf


class OverflowToolsetConfig(vf.ToolsetConfig):
    payload_chars: int = Field(65_536, gt=0)


class OverflowToolset(vf.Toolset[OverflowToolsetConfig]):
    TOOL_PREFIX = "overflow"

    def __init__(self, config: OverflowToolsetConfig):
        super().__init__(config)
        self.called = False

    @vf.tool
    def overflow_context(self) -> str:
        """Return a payload that is intentionally larger than the model context."""
        if self.called:
            return "The overflow already occurred. Answer `recovered` now."
        self.called = True
        block = "0123456789abcdef "
        repeats = self.config.payload_chars // len(block) + 1
        return (block * repeats)[: self.config.payload_chars]


class ContextCompactionTaskConfig(vf.TaskConfig):
    scenario: Literal["decode", "tool_result"] = "decode"
    payload_chars: int = Field(65_536, gt=0)
    tools: OverflowToolsetConfig = OverflowToolsetConfig()


class ContextCompactionTask(
    vf.Task[vf.TaskData, vf.State, ContextCompactionTaskConfig]
):
    @classmethod
    def toolsets(cls, config: ContextCompactionTaskConfig) -> list[vf.Toolset]:
        if config.scenario != "tool_result":
            return []
        tool_config = config.tools.model_copy(
            update={"payload_chars": config.payload_chars}
        )
        return [OverflowToolset(tool_config)]

    @vf.reward
    async def compacted(self, trace: vf.Trace) -> float:
        return float(trace.num_branches > 1)


class ContextCompactionConfig(vf.TasksetConfig):
    task: ContextCompactionTaskConfig = ContextCompactionTaskConfig()


class ContextCompactionTaskset(
    vf.Taskset[ContextCompactionTask, ContextCompactionConfig]
):
    def load(self) -> list[ContextCompactionTask]:
        if self.config.task.scenario == "decode":
            prompt = (
                "Write `x ` repeatedly. Do not use tools and do not stop. "
                "Continue until the model context ends the decode."
            )
        else:
            prompt = (
                "Call the `overflow_context` tool exactly once, then answer `recovered`. "
                "In an RLM IPython session, call it with "
                "`result = await overflow_overflow_context(); print(result)`."
            )
        return [
            ContextCompactionTask(
                vf.TaskData(idx=0, prompt=prompt),
                self.config.task,
            )
        ]


__all__ = ["ContextCompactionTaskset"]


if __name__ == "__main__":
    OverflowToolset(OverflowToolsetConfig()).run()
