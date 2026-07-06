import asyncio
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch

from verifiers.utils.save_utils import make_serializable
from verifiers.v1.clients import Client, RolloutContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.env import Environment
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import SamplingConfig

if TYPE_CHECKING:
    from verifiers.gepa.display import GEPADisplay

logger = logging.getLogger(__name__)


class V1GEPARow(TypedDict):
    example_id: int
    task: Task


def make_v1_gepa_dataset(
    tasks: list[Task], n: int, seed: int, example_offset: int = 0
) -> list[V1GEPARow]:
    """Select v1 tasks for GEPA. Repeats tiny tasksets so GEPA's requested count is honored."""
    if not tasks:
        return []

    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    selected = (
        shuffled if n < 0 else [shuffled[idx % len(shuffled)] for idx in range(n)]
    )
    return [
        {"example_id": example_offset + idx, "task": task}
        for idx, task in enumerate(selected)
    ]


def shared_initial_prompt_v1(tasks: list[Task]) -> str:
    prompts = [task.system_prompt or "" for task in tasks]
    initial_prompt = prompts[0] if prompts else ""
    if len(set(prompts)) > 1:
        logger.warning(
            "Multiple v1 task system prompts detected; GEPA will optimize one "
            "shared prompt initialized from the first task."
        )
    return initial_prompt


def ensure_v1_gepa_supported(env: Environment) -> None:
    if discover_decorated(env.taskset, "group_reward"):
        raise ValueError("v1 GEPA does not support tasksets with @group_reward yet.")


@dataclass
class VerifiersV1GEPAAdapter:
    """Bridges GEPA optimization with native v1 Environment/Episode execution."""

    env: Environment
    client: Client
    model: str
    sampling: SamplingConfig
    runner: asyncio.Runner
    max_concurrent: int = 32
    state_columns: list[str] = field(default_factory=list)
    display: "GEPADisplay | None" = None

    # GEPA adapter protocol: None means use default proposer with reflection_lm.
    propose_new_texts: Callable[..., dict[str, str]] | None = None

    _seen_prompts: dict[str, int] = field(default_factory=dict)

    def evaluate(
        self,
        batch: list[V1GEPARow],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        outputs = self.runner.run(self._evaluate(batch, candidate))
        example_ids = [output["example_id"] for output in outputs]
        rewards = [output["reward"] for output in outputs]

        if self.display is not None:
            prompt_text = candidate.get("system_prompt", "")
            if prompt_text not in self._seen_prompts:
                self._seen_prompts[prompt_text] = len(self._seen_prompts)
            self.display.update_eval(
                candidate_idx=self._seen_prompts[prompt_text],
                scores=rewards,
                example_ids=example_ids,
                capture_traces=capture_traces,
            )

        return EvaluationBatch(
            outputs=outputs,
            scores=rewards,
            trajectories=outputs if capture_traces else None,
        )

    async def _evaluate(
        self, batch: list[V1GEPARow], candidate: dict[str, str]
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        system_prompt = candidate.get("system_prompt", "")
        ctx = RolloutContext(
            client=self.client,
            model=self.model,
            sampling=self.sampling,
        )

        async def run_one(row: V1GEPARow) -> dict[str, Any]:
            task = row["task"].model_copy(update={"system_prompt": system_prompt})
            episode = self.env.episode(task, ctx, n=1)
            (trace,) = await episode.run(semaphore)
            return self._trace_output(row, trace)

        return await asyncio.gather(*(run_one(row) for row in batch))

    def _trace_output(self, row: V1GEPARow, trace: Trace) -> dict[str, Any]:
        error = trace.error.model_dump(mode="json") if trace.error else None
        task_dump = trace.task.model_dump(mode="json")
        output: dict[str, Any] = {
            "example_id": row["example_id"],
            "prompt": trace.task.prompt_text,
            "completion": trace.last_reply,
            "transcript": trace.transcript,
            "answer": task_dump.get("answer", ""),
            "reward": trace.reward,
            "rewards": dict(trace.rewards),
            "metrics": dict(trace.metrics),
            "stop_condition": trace.stop_condition,
            "error": error,
            "task": task_dump,
            "trace": trace.to_record(),
        }
        for col in self.state_columns:
            if col in trace.info:
                output[col] = make_serializable(trace.info[col])
            elif col in trace.metrics:
                output[col] = trace.metrics[col]
            elif col in trace.rewards:
                output[col] = trace.rewards[col]
        return output

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],  # noqa: ARG002 - required by GEPA adapter protocol
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records: list[dict[str, Any]] = []
        trajectories = eval_batch.trajectories or eval_batch.outputs
        for output, trajectory, score in zip(
            eval_batch.outputs, trajectories, eval_batch.scores
        ):
            record: dict[str, Any] = {
                "query": output["prompt"],
                "completion": output["completion"],
                "transcript": output["transcript"],
                "expected_answer": output.get("answer", ""),
                "reward": score,
                "rewards": output.get("rewards", {}),
                "metrics": output.get("metrics", {}),
            }
            if trajectory.get("error"):
                record["error"] = trajectory["error"]
            if trajectory.get("stop_condition"):
                record["stop_condition"] = trajectory["stop_condition"]
            for col in self.state_columns:
                if col in output:
                    record[col] = make_serializable(output[col])
            records.append(record)

        return {comp: records for comp in components_to_update}
