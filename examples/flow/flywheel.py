"""Offline partial-evaluation recovery, using deterministic substitutes for expensive work.

    uv run python -m examples.flow.flywheel ./out
    uv run python -m verifiers.v1.flow inspect ./out
    uv run python -m verifiers.v1.flow steer ./out task --status ready
    uv run python -m examples.flow.flywheel ./out --available

The first run records six answers and holds on two failures. After the operator releases
it, the second run executes only those two. Changing the declared task inputs reruns all
slots. Provider availability is operational and deliberately absent from reuse inputs.
"""

import argparse
import asyncio
from pathlib import Path

from pydantic import BaseModel

from verifiers.v1.flow import (
    Ctx,
    Flow,
    FlowConfig,
    Pipeline,
    Transition,
    UnitData,
    drain_on_interrupt,
    fn,
)


class TaskData(UnitData):
    question: str = "What is 6 * 7?"
    expected: str = "42"


class Evaluation(BaseModel):
    answer: str
    score: float


async def solve(task: TaskData, slot: int, available: bool) -> Evaluation:
    print(f"executing solver {slot}")
    if slot >= 6 and not available:
        raise RuntimeError("provider unavailable")
    return Evaluation(answer="42", score=float(task.expected == "42"))


async def run(root: Path, available: bool) -> None:
    async def evaluate(ctx: Ctx[TaskData]) -> Transition:
        results = await ctx.spread(
            [
                fn(solve, ctx.data, i, available, output=Evaluation, inputs=ctx.data)
                for i in range(8)
            ],
            key=lambda i: f"solve/{i}",
        )
        if failures := [r.error for r in results if not r.ok]:
            return Transition.hold(f"{len(failures)}/8 failed: {failures}")
        return Transition.end(
            "evaluated",
            f"8 completed; score {sum(r.value.score for r in results if r.value is not None) / 8}",
        )

    async with Flow(root, FlowConfig(), Pipeline({"evaluate": evaluate})) as flow:
        flow.create_unit("task", stage="evaluate", data=TaskData())
        drain_on_interrupt(flow)
        print((await flow.run()).model_dump_json(indent=2))


if __name__ == "__main__":
    from examples.flow.flywheel import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--available", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.root, args.available))
