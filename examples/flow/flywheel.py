"""Offline partial-evaluation recovery, using deterministic substitutes for expensive work.

With Prime-RL installed:
    uv run flow run examples.flow.flywheel.entrypoint ./out
    uv run flow inspect ./out
    uv run flow steer ./out task --status ready
    uv run flow run examples.flow.flywheel.entrypoint ./out --available

The first run records six answers and holds on two failures. After the operator releases
it, the second run executes only those two. Changing the declared task inputs reruns all
slots. Provider availability is operational and deliberately absent from reuse inputs.

For a monitored process, use FlowConfig(stay_alive=True): idle units can be steered
without relaunching; the drain command ends the run. Data and status can be changed
together with `steer --data patch.json --expected <revision> --status ready`.

Flow seeds pools.json from config once, then checks for edits about every two seconds.
Replace the complete mapping atomically (same pool names, positive integer limits):
    printf '%s\\n' '{"units": 2, "runtimes": 4}' > ./out/pools.json.tmp
    mv ./out/pools.json.tmp ./out/pools.json
Limits persist across launches. Lowering them lets admitted work finish; malformed
updates log a warning and leave the last valid limits in effect.
Admission rules can read the effective limits from flow.pools.limits.
"""

from pathlib import Path

from pydantic import BaseModel

from verifiers.v1.flow import (
    Ctx,
    Flow,
    FlowConfig,
    FlowEntrypoint,
    Pipeline,
    Transition,
    UnitData,
    drain_on_interrupt,
    fn,
)


class TaskData(UnitData):
    question: str = "What is 6 * 7?"
    expected: str = "42"


class Config(FlowConfig):
    available: bool = False


class Evaluation(BaseModel):
    answer: str
    score: float


async def solve(task: TaskData, slot: int, available: bool) -> Evaluation:
    print(f"executing solver {slot}")
    if slot >= 6 and not available:
        raise RuntimeError("provider unavailable")
    return Evaluation(answer="42", score=float(task.expected == "42"))


async def run(root: Path, config: Config) -> int:
    async def evaluate(ctx: Ctx[TaskData]) -> Transition[TaskData]:
        results = await ctx.spread(
            [
                fn(
                    solve,
                    ctx.data,
                    i,
                    config.available,
                    output=Evaluation,
                    inputs=ctx.data,
                )
                for i in range(8)
            ],
            key=lambda i: f"solve/{i}",
        )
        if failures := [r.error.message for r in results if not r.ok]:
            return Transition(
                "held", f"{len(failures)}/8 failed: {failures}", status="held"
            )
        return Transition(
            "evaluated",
            f"8 completed; score {sum(r.value.score for r in results if r.ok) / 8}",
            status="terminal",
        )

    async with Flow(root, config, Pipeline({"evaluate": evaluate})) as flow:
        flow.create_unit("task", stage="evaluate", data=TaskData())
        drain_on_interrupt(flow)
        result = await flow.run()
        print(result.model_dump_json(indent=2))
        return 0 if result.counts == {"terminal": 1} else 1


entrypoint = FlowEntrypoint(Config, run)
