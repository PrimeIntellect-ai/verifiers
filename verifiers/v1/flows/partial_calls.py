"""Offline example: six calls finish, two fail; releasing retries only those two."""

from verifiers.v1.flow import Flow, FlowConfig, Transition, Unit, UnitData, stage


class Data(UnitData):
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
        self.create_unit("task", stage="evaluate", data=Data())

    @stage
    async def evaluate(self, unit: Unit[Data]) -> Transition[Data]:
        results = await self.gather(
            *(
                self.attempt(
                    solve,
                    unit.data,
                    i,
                    self.config.available,
                    output=float,
                    key=f"solve/{i}",
                    cache_inputs=unit.data.model_dump(mode="json"),
                )
                for i in range(8)
            )
        )
        if failures := [r.error.message for r in results if not r.ok]:
            return Transition(
                "held", f"{len(failures)}/8 failed: {failures}", status="held"
            )
        return Transition(
            "evaluated",
            f"mean score {sum(r.value for r in results if r.ok) / 8}",
            status="terminal",
        )


__all__ = ["PartialCalls"]
