"""One independently resumable unit per prompt."""

from pydantic import Field

from verifiers.v1 import AgentConfig, SubprocessConfig, Task, TaskData
from verifiers.v1.flow import Flow, FlowConfig, Transition, Unit, UnitData, stage


class Config(FlowConfig):
    solver: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    prompts: list[str] = Field(default_factory=lambda: ["What is 2 + 2?"])


class Data(UnitData):
    prompt: str


class SingleAgent(Flow[Config]):
    async def setup(self) -> None:
        for i, prompt in enumerate(self.config.prompts):
            self.create_unit(str(i), stage="solve", data=Data(prompt=prompt))

    @stage
    async def solve(self, unit: Unit[Data]) -> Transition[Data]:
        trace = await self.agents.solver.run(Task(TaskData(prompt=unit.data.prompt)))
        return Transition("answered", trace.last_reply, status="terminal")


__all__ = ["SingleAgent"]
