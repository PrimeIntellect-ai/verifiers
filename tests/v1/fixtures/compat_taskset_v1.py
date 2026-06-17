"""Taskset fixture for v1-as-v0 compatibility tests."""

from __future__ import annotations

import verifiers.v1 as vf


class CompatTask(vf.Task):
    answer: str


class CompatTasksetConfig(vf.TasksetConfig):
    id: str = "compat-taskset-v1"
    phrase: str = "alpha"


class CompatTaskset(vf.Taskset[CompatTask, CompatTasksetConfig]):
    def load_tasks(self) -> list[CompatTask]:
        return [
            CompatTask(
                idx=0,
                system_prompt="Return the requested phrase.",
                instruction=self.config.phrase,
                answer=self.config.phrase,
            )
        ]

    @vf.reward
    async def answered(self, task: CompatTask, trace: vf.Trace) -> float:
        text = trace.assistant_messages[-1].content if trace.assistant_messages else ""
        return float(task.answer in (text or ""))

    @vf.metric
    async def turns(self, trace: vf.Trace) -> float:
        return float(trace.num_turns)


def load_taskset(config: CompatTasksetConfig) -> CompatTaskset:
    return CompatTaskset(config)
