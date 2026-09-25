"""Offline example: six calls finish, two fail; releasing retries only those two."""

from verifiers.v1.flow import Flow, FlowConfig, Job, JobData, Transition, stage


class Data(JobData):
    question: str = "What is 6 * 7?"
    expected: str = "42"


class Config(FlowConfig):
    available: bool = False


async def solve(task: Data, slot: int, available: bool) -> float:
    print(f"executing solver {slot}")
    if slot >= 6 and not available:
        raise RuntimeError("provider unavailable")
    return float(task.expected == "42")


class PartialCalls(Flow[Config]):
    async def setup(self) -> None:
        self.create("task", stage="evaluate", data=Data())

    @stage
    async def evaluate(self, job: Job[Data]) -> Transition[Data]:
        results = await self.gather(
            *(
                self.attempt(
                    solve,
                    job.data,
                    i,
                    self.config.available,
                    output=float,
                    key=f"solve/{i}",
                    cache_inputs=job.data.model_dump(mode="json"),
                )
                for i in range(8)
            )
        )
        if failures := [r.error.message for r in results if not r.ok]:
            return Transition(
                outcome="held",
                reason=f"{len(failures)}/8 failed: {failures}",
                status="held",
            )
        return Transition(
            outcome="evaluated",
            reason=f"mean score {sum(r.value for r in results if r.ok) / 8}",
            status="terminal",
        )


__all__ = ["PartialCalls"]
