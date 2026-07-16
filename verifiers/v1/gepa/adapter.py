"""The GEPA <-> v1 bridge: run a candidate system prompt over a batch of tasks and score it.

GEPA's adapter protocol (`evaluate`, `make_reflective_dataset`) is synchronous and
`gepa.api.optimize` blocks, but v1 rollouts are async. The runner manages one event loop by
hand: it enters `env.serving()` on that loop and runs the blocking `optimize()` on the main
thread, so each synchronous `evaluate()` drives its batch of rollouts with
`loop.run_until_complete` — the one sync↔async hop. (Mirrors how v0 vf-gepa bridged; no worker
thread, so a Ctrl-C unwinds straight through `optimize()` into the runner's teardown.)
"""

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from gepa.core.adapter import EvaluationBatch
from pydantic_core import to_jsonable_python

from verifiers.v1.clients import ModelContext
from verifiers.v1.env import Environment
from verifiers.v1.task import Task
from verifiers.v1.trace import RolloutRecord, Trace

Candidate = dict[str, str]


@dataclass
class GEPAAdapter:
    """Bridges GEPA's optimization loop with a native v1 `Environment`. `tasks` covers only the
    trainset + valset tasks GEPA was given (not the whole taskset), keyed by `task.data.idx` —
    GEPA's `batch` is a list of those idxs, and injecting the candidate rebuilds each `Task`
    around a `data` row carrying the new `system_prompt`. `loop` is the runner's persistent event
    loop (which holds `env.serving()` open); `evaluate` drives its rollouts on it with
    `run_until_complete`."""

    env: Environment
    ctx: ModelContext
    tasks: dict[int, Task]
    loop: asyncio.AbstractEventLoop
    semaphore: asyncio.Semaphore | None = None
    on_complete: Callable[[RolloutRecord], Awaitable[None]] | None = None
    """Called with each rollout's record as it finalizes — the runner's persist hook that
    streams records to `traces.jsonl`, exactly as `run_eval` does."""
    reflection_columns: list[str] = field(default_factory=list)
    propose_new_texts: Callable[..., Candidate] | None = None
    """Part of GEPA's adapter protocol — its proposer reads this attribute on every reflection
    step. None = use GEPA's default reflection-LM proposer (the AttributeError from leaving it
    undeclared silently disables all mutation proposals)."""

    def evaluate(
        self,
        batch: list[int],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trace, Trace]:
        """Run `candidate`'s system prompt on the tasks named by `batch` (`Task.idx` values)
        and score them. Called synchronously by GEPA on the main thread; each batch's rollouts
        run on the runner's persistent loop via `run_until_complete`."""
        system_prompt = candidate.get("system_prompt", "")
        traces = self.loop.run_until_complete(self._run_batch(batch, system_prompt))
        scores = [trace.reward for trace in traces]
        return EvaluationBatch(
            outputs=traces,
            scores=scores,
            trajectories=traces if capture_traces else None,
        )

    async def _run_batch(self, batch: list[int], system_prompt: str) -> list[Trace]:
        # Inject the candidate by rebuilding each Task around a data row with the new
        # system_prompt (TaskData is frozen; behavior/config carry over unchanged).
        tasks = [
            type(t)(
                t.data.model_copy(update={"system_prompt": system_prompt}), t.config
            )
            for t in (self.tasks[idx] for idx in batch)
        ]
        episodes = [self.env.episode(task, self.ctx, n=1) for task in tasks]
        results = await asyncio.gather(
            *(episode.run(self.semaphore, self.on_complete) for episode in episodes)
        )
        return [
            trace
            for episode_records in results
            for record in episode_records
            for trace in record.traces
        ]

    def make_reflective_dataset(
        self,
        candidate: Candidate,  # noqa: ARG002 - required by GEPA's adapter protocol
        eval_batch: EvaluationBatch[Trace, Trace],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build the reflective dataset the teacher LM reads to propose a new system prompt,
        from `eval_batch.trajectories` (traces captured by a prior `evaluate(capture_traces=True)`
        on the same batch)."""
        traces = eval_batch.trajectories or []
        records = []
        for trace, score in zip(traces, eval_batch.scores):
            record: dict[str, Any] = {
                "query": trace.task.data.prompt_text,
                "completion": trace.last_reply,
                "reward": score,
            }
            if trace.has_error:
                record["error"] = str(trace.error)
            if trace.stop_condition:
                record["stop_condition"] = trace.stop_condition
            for column in self.reflection_columns:
                if column in trace.info:
                    record[column] = to_jsonable_python(trace.info[column])
                elif hasattr(trace.task.data, column):
                    record[column] = to_jsonable_python(
                        getattr(trace.task.data, column)
                    )
            records.append(record)
        return {comp: records for comp in components_to_update}
