"""The replay-buffer taskset: resume old rollouts from tagged compaction points.

Two sourcing modes (``config.mode``):

- ``offline`` — ``load_tasks`` materializes one task per compaction resume point found in the
  buffer at env-server start (snapshot of a prior run's, or this run's, rollouts). The replay
  prefix (``root->node``) is the task ``prompt``; run with the default harness.
- ``online`` — ``load_tasks`` returns ``pool_size`` virtual slots and the bundled
  :class:`ReplayHarness` samples a stored trace + resume point from the *live* buffer at rollout
  time, so a growing buffer (this run's own rollouts) is picked up as it fills.

Each compaction exposes two resume points:
- ``compaction_after``  -> prefix ends at the compaction message; the model continues solving.
- ``compaction_before`` -> prefix is the pre-compaction context; the model writes the compaction
  itself, then continues.

Scoring reuses the ORIGINAL env's verifier: ``config.inner`` names the taskset the rollouts came
from; ``score`` runs its ``@reward``/``@metric`` over the replay trace with the original task
swapped in (so e.g. a math verifier checks the replayed continuation's final answer against the
original ground truth). The original task + reward ride on the replay task (offline) or in
``trace.info["replay"]`` (online).
"""

from __future__ import annotations

import glob
import json
from typing import Literal

from pydantic import SerializeAsAny

from verifiers.v1.loaders import load_taskset
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, WireTask
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace, WireTrace

from replay.selector import resume_points, seed_messages, snapshot_ref_of


class ReplayTask(Task):
    """A replayed prefix plus the provenance needed to score it with the original verifier
    (offline mode; online mode carries the same provenance in ``trace.info["replay"]``)."""

    source_trace_id: str = ""
    resume_node: int = -1
    resume_kind: str = ""
    snapshot_ref: str | None = None
    original_task: dict = {}
    original_reward: float = 0.0


class ReplayTasksetConfig(TasksetConfig):
    mode: Literal["offline", "online"] = "offline"
    buffer_glob: str = ""
    """Glob of stored-rollout JSONL files, e.g. ``.../rollouts/step_*/train_rollouts.jsonl``."""
    kinds: list[str] = ["compaction_after", "compaction_before"]
    """Which resume-point kinds to replay from."""
    pool_size: int = 1024
    """Online mode: number of virtual task slots (num_tasks); the harness samples per rollout."""
    inner: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The ORIGINAL env's taskset (the verifier to reuse). Empty id => scoring is a no-op."""


def _iter_traces(buffer_glob: str):
    for path in sorted(glob.glob(buffer_glob)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield WireTrace.model_validate(json.loads(line))


class ReplayTaskset(Taskset[ReplayTask, ReplayTasksetConfig]):
    def __init__(self, config: ReplayTasksetConfig) -> None:
        super().__init__(config)
        # Reuse the original env's verifier when configured (skeleton: empty id => no scoring).
        self._inner: Taskset | None = load_taskset(config.inner) if config.inner.id else None

    def load_tasks(self) -> list[ReplayTask]:
        if self.config.mode == "online":
            # Virtual slots; ReplayHarness samples the live buffer per rollout.
            return [ReplayTask(idx=i, prompt=None) for i in range(self.config.pool_size)]
        kinds = set(self.config.kinds)
        tasks: list[ReplayTask] = []
        for src in _iter_traces(self.config.buffer_glob):
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
        # Offline exec/sandbox replay: restore to the resume point before the harness runs.
        # (Online restores inside ReplayHarness, where the sampled ref is known.) Skeleton:
        # snapshot capture isn't wired yet, so refs are None -> nothing to restore.
        if task.snapshot_ref is not None:
            await runtime.restore(task.snapshot_ref)

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        # Reuse the ORIGINAL env's verifier: run its rewards/metrics over the replay trace with
        # the original task swapped in. (Judge-style scoring would instead compare the model's
        # verdict against ``original_reward`` here.)
        if self._inner is None:
            return
        replay = trace.info.get("replay")  # online: harness-stashed provenance
        original_dump = replay["original_task"] if replay else getattr(trace.task, "original_task", {})
        if not original_dump:
            return
        replay_task = trace.task
        trace.task = WireTask.model_validate(original_dump)
        try:
            await self._inner.score(trace, runtime)
        finally:
            trace.task = replay_task


# Bundle the online sampling harness so `--harness.id replay` resolves alongside the taskset.
from replay.harness import ReplayHarness, ReplayHarnessConfig  # noqa: E402

__all__ = [
    "ReplayTaskset",
    "ReplayTasksetConfig",
    "ReplayTask",
    "ReplayHarness",
    "ReplayHarnessConfig",
]
