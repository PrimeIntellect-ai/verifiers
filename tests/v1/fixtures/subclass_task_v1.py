"""A taskset whose second case is a per-case `Task` subclass carrying its own reward and
toolset — the shape a served run cannot carry over the wire."""

import verifiers.v1 as vf


class CaseData(vf.TaskData):
    answer: str


class ExtraToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "extra"

    @vf.tool
    def ping(self) -> str:
        """Reply pong."""
        return "pong"


class BaseTask(vf.Task[CaseData]):
    @vf.reward
    async def base_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == self.data.answer)


class SpecialTask(BaseTask):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [ExtraToolset(vf.ToolsetConfig())]

    @vf.reward
    async def special_match(self, trace: vf.Trace) -> float:
        return 1.0


class SubclassTaskset(vf.Taskset[BaseTask, vf.TasksetConfig]):
    def load(self) -> list[BaseTask]:
        return [
            BaseTask(CaseData(idx=0, prompt="say a", answer="a"), self.config.task),
            SpecialTask(CaseData(idx=1, prompt="say b", answer="b"), self.config.task),
        ]


__all__ = ["SubclassTaskset"]
