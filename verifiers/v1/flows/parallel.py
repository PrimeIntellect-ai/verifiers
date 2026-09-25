"""Keep finished solvers when a parallel sibling fails; release the job to retry."""

from pydantic import PositiveInt

from verifiers.v1 import AgentConfig, SubprocessConfig, Task, TaskData
from verifiers.v1.flow import Flow, FlowConfig, Job, JobData, Transition, stage


class Config(FlowConfig):
    solver: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    prompt: str = "What is 2 + 2?"
    samples: PositiveInt = 4


class Data(JobData):
    prompt: str


class Parallel(Flow[Config]):
    async def setup(self) -> None:
        self.create("task", stage="solve", data=Data(prompt=self.config.prompt))

    @stage
    async def solve(self, job: Job[Data]) -> Transition[Data]:
        solver = self.agents.solver
        # This pipeline chooses its reuse dependencies; execution budgets are excluded.
        cache_inputs = {
            "prompt": job.data.prompt,
            "agent": solver.config.model_dump(
                mode="json",
                include={
                    "model": True,
                    "sampling": True,
                    "harness": True,
                    "client": {"type", "base_url", "renderer"},
                },
            ),
        }
        await self.gather(
            *(
                solver.run(
                    Task(TaskData(prompt=job.data.prompt)),
                    key=str(i),
                    cache_inputs=cache_inputs,
                )
                for i in range(self.config.samples)
            )
        )
        return Transition(outcome="evaluated", status="terminal")


__all__ = ["Parallel"]
