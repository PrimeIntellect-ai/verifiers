"""Two stages exchange an immutable document revision, independent of workflow state."""

from verifiers.v1 import AgentConfig, SubprocessConfig, Task, TaskData
from verifiers.v1.flow import (
    Flow,
    FlowConfig,
    GitArtifacts,
    Job,
    JobData,
    Transition,
    stage,
)


class Config(FlowConfig):
    writer: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    reviewer: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    prompt: str = "Write a short explanation of how rain forms."


class Data(JobData):
    prompt: str
    revision: str | None = None


class DraftReview(Flow[Config]):
    async def setup(self) -> None:
        self.create("document", stage="draft", data=Data(prompt=self.config.prompt))

    @stage
    async def draft(self, job: Job[Data]) -> Transition[Data]:
        trace = await self.agents.writer.run(Task(TaskData(prompt=job.data.prompt)))
        job.data.revision = GitArtifacts(job.path).write(
            base=None, files={"draft.md": trace.last_reply}
        )
        return Transition(outcome="written", stage="review", data=job.data)

    @stage
    async def review(self, job: Job[Data]) -> Transition[Data]:
        assert job.data.revision is not None
        draft = GitArtifacts(job.path).read(job.data.revision, "draft.md")
        trace = await self.agents.reviewer.run(
            Task(TaskData(prompt=f"Review this draft:\n{draft}"))
        )
        return Transition(
            outcome="reviewed", reason=trace.last_reply, status="terminal"
        )


__all__ = ["DraftReview"]
