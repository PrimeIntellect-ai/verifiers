"""Keep finished solvers when a parallel sibling fails; release the unit to retry."""

from verifiers.v1 import AgentConfig, SubprocessConfig, Task, TaskData
from verifiers.v1.flow import Flow, FlowConfig, Transition, Unit, UnitData, stage


class Config(FlowConfig):
    solver: AgentConfig = AgentConfig(runtime=SubprocessConfig())
    prompt: str = "What is 2 + 2?"
    samples: int = 4


class Data(UnitData):
    prompt: str


class Parallel(Flow[Config]):
    async def setup(self) -> None:
        self.create_unit("task", stage="solve", data=Data(prompt=self.config.prompt))

    @stage
    async def solve(self, unit: Unit[Data]) -> Transition[Data]:
        solver = self.agents.solver
        # This pipeline chooses its reuse dependencies; execution budgets are excluded.
        inputs = {
            "prompt": unit.data.prompt,
            "agent": solver.config.model_dump(
                mode="json", include={"model", "sampling", "harness"}
            ),
        }
        results = await self.gather(
            *(
                solver.attempt(
                    Task(TaskData(prompt=unit.data.prompt)), key=str(i), inputs=inputs
                )
                for i in range(self.config.samples)
            )
        )
        failures = [r.error.message for r in results if not r.ok]
        return Transition(
            "evaluated", "\n".join(failures), status="held" if failures else "terminal"
        )


__all__ = ["Parallel"]
