"""A minimal replay-buffer taskset (skeleton).

Materializes replay tasks from a buffer of stored rollouts (offline mode): each compaction
resume point in each stored trace becomes one task whose ``prompt`` is the replay prefix
(root->resume node). Run it with the default harness (``--harness.id default``), which accepts
a Messages prompt and continues the rollout from the prefix:

- ``compaction_after``  -> the prefix ends at the compaction message; the model continues solving.
- ``compaction_before`` -> the prefix is the pre-compaction context; the model first writes the
  compaction (the compact system prompt instructs it), then continues.

Scoring reuses the ORIGINAL task's verifier (skeleton stub below). Exec/sandbox envs restore the
resume point's snapshot in ``setup`` before the harness runs (skeleton: ref is None -> no-op).

NOTE (online/growing buffer): ``load_tasks`` runs once at env-server start, so this offline
materialization sees only the buffer present at startup. The online variant samples a stored
trace + resume point per rollout in a custom replay harness (rollout-time), keeping num_tasks
virtual; left as a follow-up.
"""

from __future__ import annotations

import glob
import json

from verifiers.v1.decorators import reward
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace, WireTrace

from replay.selector import resume_points, seed_messages, snapshot_ref_of


class ReplayTask(Task):
    """A replayed prefix plus the provenance needed to score it with the original verifier."""

    source_trace_id: str = ""
    resume_node: int = -1
    resume_kind: str = ""
    snapshot_ref: str | None = None
    original_task: dict = {}
    original_reward: float = 0.0


class ReplayTasksetConfig(TasksetConfig):
    buffer_glob: str = ""
    """Glob of stored-rollout JSONL files, e.g. ``.../rollouts/step_*/train_rollouts.jsonl``."""
    kinds: list[str] = ["compaction_after", "compaction_before"]
    """Which resume-point kinds to materialize tasks from."""


class ReplayTaskset(Taskset[ReplayTask, ReplayTasksetConfig]):
    def load_tasks(self) -> list[ReplayTask]:
        tasks: list[ReplayTask] = []
        kinds = set(self.config.kinds)
        for path in sorted(glob.glob(self.config.buffer_glob)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    src: Trace = WireTrace.model_validate(json.loads(line))
                    for pt in resume_points(src, kinds=kinds):
                        tasks.append(
                            ReplayTask(
                                idx=len(tasks),
                                prompt=seed_messages(src, pt["node"]),
                                source_trace_id=src.id,
                                resume_node=pt["node"],
                                resume_kind=pt["kind"],
                                snapshot_ref=snapshot_ref_of(src, pt["node"]),
                                original_task=src.task.model_dump(),
                                original_reward=src.reward,
                            )
                        )
        return tasks

    async def setup(self, task: ReplayTask, runtime: Runtime) -> None:
        # Exec/sandbox replay: restore the sandbox to the resume point before the harness runs.
        # Skeleton: snapshot capture isn't wired yet, so ref is None -> nothing to restore.
        if task.snapshot_ref is not None:
            await runtime.restore(task.snapshot_ref)

    @reward
    def replay_reward(self, trace: Trace, task: ReplayTask) -> float:
        # TODO: reuse the ORIGINAL env's verifier — reconstruct its taskset + the original Task
        # from ``task.original_task`` and delegate (``inner.score(trace, runtime)``); for the
        # judge-style mode, compare the model's verdict against ``task.original_reward``.
        # Skeleton placeholder so the env is runnable end-to-end.
        return 0.0


__all__ = ["ReplayTaskset", "ReplayTasksetConfig", "ReplayTask"]
