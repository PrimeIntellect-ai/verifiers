"""Two stages exchange an immutable document revision, independent of workflow state."""

from verifiers.v1 import AgentConfig, SubprocessConfig, Task, TaskData
from verifiers.v1.flow import (
    Flow,
    FlowConfig,
    GitArtifacts,
    Transition,
    Unit,
    UnitData,
    stage,
)


class Config(FlowConfig):
    writer: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    reviewer: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    prompt: str = "Write a short explanation of how rain forms."


class Data(UnitData):
    prompt: str
    revision: str | None = None


class DraftReview(Flow[Config]):
    async def setup(self) -> None:
        self.create_unit(
            "document", stage="draft", data=Data(prompt=self.config.prompt)
        )

    @stage
    async def draft(self, unit: Unit[Data]) -> Transition[Data]:
        trace = await self.agents.writer.run(Task(TaskData(prompt=unit.data.prompt)))
        unit.data.revision = GitArtifacts(unit).write(
            base=None, files={"draft.md": trace.last_reply}
        )
        return Transition("written", stage="review", data=unit.data)

    @stage
    async def review(self, unit: Unit[Data]) -> Transition[Data]:
        assert unit.data.revision is not None
        draft = GitArtifacts(unit).read(unit.data.revision, "draft.md")
        trace = await self.agents.reviewer.run(
            Task(TaskData(prompt=f"Review this draft:\n{draft}"))
        )
        return Transition("reviewed", trace.last_reply, status="terminal")


__all__ = ["DraftReview"]
