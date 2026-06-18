"""Group-scored taskset fixture for v1-as-v0 compatibility tests."""

from __future__ import annotations

import verifiers.v1 as vf


class CompatGroupTask(vf.Task):
    answer: str


class CompatGroupTasksetConfig(vf.TasksetConfig):
    id: str = "compat-group-taskset-v1"


class CompatGroupTaskset(vf.Taskset[CompatGroupTask, CompatGroupTasksetConfig]):
    def load_tasks(self) -> list[CompatGroupTask]:
        return [
            CompatGroupTask(
                idx=0,
                prompt="group-alpha",
                answer="group-alpha",
            )
        ]

    @vf.reward
    async def base(self, trace: vf.Trace) -> float:
        return float(bool(trace.assistant_messages))

    @vf.group_reward
    async def rank(self, traces: list[vf.Trace]) -> list[float]:
        return [float(i + 1) for i, _trace in enumerate(traces)]


def load_taskset(config: CompatGroupTasksetConfig) -> CompatGroupTaskset:
    return CompatGroupTaskset(config)
