"""Thin task rows over an installed, pinned Tau distribution."""

import os
from typing import ClassVar, Literal

from pydantic import Field

import verifiers.v1 as vf
from verifiers.v1.tasksets.tau.data import prepare_data


class TauConfig(vf.TasksetConfig):
    domain: str = "telecom"
    split: str = "base"
    max_steps: int | None = Field(None, gt=0)
    retrieval_config: str | None = None


class TauData(vf.TaskData):
    tau_task: dict
    domain: str
    repository: str
    revision: str
    runner: Literal["upstream", "synth"] = "upstream"
    max_steps: int | None = None
    retrieval_config: str | None = None


class TauTask(vf.Task[TauData]):
    @vf.reward
    async def tau_reward(self, trace: vf.Trace) -> float:
        return float(trace.info["tau"]["reward_info"]["reward"])

    @vf.metric
    async def tau_metrics(self, trace: vf.Trace) -> dict[str, float]:
        messages = trace.info["tau"]["messages"]
        return {
            "num_steps": float(len(messages)),
            "num_errors": float(sum(bool(m.get("error")) for m in messages)),
            **{
                f"num_{role}_tool_calls": float(
                    sum(
                        len(m.get("tool_calls") or [])
                        for m in messages
                        if m["role"] == role
                    )
                )
                for role in ("assistant", "user")
            },
        }


class TauTaskset(vf.Taskset[TauTask, TauConfig]):
    repository: ClassVar[str] = "sierra-research/tau2-bench"
    revision: ClassVar[str] = "b7ea9074c1cba482b30687fecdb5c8425fd6f619"
    runner: ClassVar[Literal["upstream", "synth"]] = "upstream"

    def load(self):
        os.environ["TAU2_DATA_DIR"] = str(prepare_data(self.repository, self.revision))
        if self.runner == "synth":
            from tau2.run import load_tasks  # ty: ignore[unresolved-import]
        else:
            from tau2.runner import load_tasks  # ty: ignore[unresolved-import]

        for index, task in enumerate(load_tasks(self.config.domain, self.config.split)):
            yield TauTask(
                TauData(
                    idx=index,
                    name=task.id,
                    prompt="",
                    tau_task=task.model_dump(mode="json"),
                    domain=self.config.domain,
                    repository=self.repository,
                    revision=self.revision,
                    runner=self.runner,
                    max_steps=self.config.max_steps,
                    retrieval_config=self.config.retrieval_config,
                ),
                self.config.task,
            )
