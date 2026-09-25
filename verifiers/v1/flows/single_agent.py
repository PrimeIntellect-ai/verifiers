"""One independently resumable job per prompt."""

from pydantic import Field

from verifiers.v1 import AgentConfig, SubprocessConfig, Task, TaskData
from verifiers.v1.flow import Flow, FlowConfig, Job, JobData, Transition, stage


class Config(FlowConfig):
    solver: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    prompts: list[str] = Field(default_factory=lambda: ["What is 2 + 2?"])


class Data(JobData):
    prompt: str


class SingleAgent(Flow[Config]):
    async def setup(self) -> None:
        for i, prompt in enumerate(self.config.prompts):
            self.create(str(i), stage="solve", data=Data(prompt=prompt))

    @stage
    async def solve(self, job: Job[Data]) -> Transition[Data]:
        trace = await self.agents.solver.run(Task(TaskData(prompt=job.data.prompt)))
        return Transition(
            outcome="answered", reason=trace.last_reply, status="terminal"
        )


__all__ = ["SingleAgent"]
