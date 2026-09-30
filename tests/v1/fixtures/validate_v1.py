"""validate: model-free fixture tasks for `vf-validate`, one per validation outcome.

Setup writes the task's untouched grading state (`state.txt`) into its runtime; the
reward grades whatever `state.txt` holds, and `validate` applies the reference answer
(writes a solved state) before grading. Every grade is logged to `GRADES`, with the
runtime it read from, so tests can see what the checks actually scored (and `RUNTIMES` which boxes
were set up). Resolved by id
`validate-v1` via pytest's `pythonpath`.
"""

import asyncio
from typing import Literal

from pydantic import Field

import verifiers.v1 as vf

GRADES: list[tuple[str, str]] = []
"""(runtime name, graded state) for every grade, in call order."""
RUNTIMES: list[str] = []
"""The runtime of every setup, in call order."""


class ValidateData(vf.TaskData):
    baseline: float = 0.0
    """The score the untouched task earns."""
    fail: Literal["setup", "score", "slow"] | None = None


class ValidateTask(vf.Task[ValidateData]):
    async def setup(self, runtime: vf.Runtime) -> None:
        RUNTIMES.append(runtime.name)
        if self.data.fail == "setup":
            raise RuntimeError("setup crashed")
        await runtime.write("state.txt", str(self.data.baseline).encode())

    async def finalize(self, trace: vf.Trace) -> None:
        trace.info["finalized"] = True

    async def grade(self, runtime: vf.Runtime) -> float:
        state = (await runtime.read("state.txt")).decode()
        GRADES.append((runtime.name, state))
        return float(state)

    @vf.reward(weight=1.0)
    async def passing(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        if not trace.info.get("finalized"):
            raise RuntimeError("scored before finalize")
        if self.data.fail == "score":
            raise RuntimeError("grader crashed")
        if self.data.fail == "slow":
            await asyncio.sleep(30)
        return await self.grade(runtime)

    async def validate(self, runtime: vf.Runtime) -> bool:
        await runtime.write("state.txt", b"1.0")
        return await self.grade(runtime) == 1.0


class UnscoredTask(ValidateTask):
    """Declares no reward, so the untouched task cannot be checked."""

    passing = None


class ValidateCasesConfig(vf.TasksetConfig):
    cases: list[str] = Field(default_factory=lambda: ["ok"])
    """Task per case: `ok`, `partial`, `trivial`, `unscored`, or a `fail` value."""


class ValidateTaskset(vf.Taskset[ValidateTask, ValidateCasesConfig]):
    def load(self) -> list[ValidateTask]:
        tasks = []
        for i, case in enumerate(self.config.cases):
            data = ValidateData(idx=i, name=case, prompt=case)
            if case in ("partial", "trivial"):
                data = data.model_copy(
                    update={"baseline": 0.5 if case == "partial" else 1.0}
                )
            elif case in ("setup", "score", "slow"):
                data = data.model_copy(update={"fail": case})
            cls = UnscoredTask if case == "unscored" else ValidateTask
            tasks.append(cls(data, self.config.task))
        return tasks


__all__ = ["ValidateTaskset"]
