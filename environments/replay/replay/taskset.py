"""The replay-buffer taskset: turn old rollouts into new training tasks.

Sourcing modes (``config.mode``):

- ``offline`` — ``load_tasks`` materializes one task per resume point found in the buffer at
  env-server start. The seed conversation is the task ``prompt``; run with the default harness.
- ``online`` — ``load_tasks`` returns ``pool_size`` virtual slots and the bundled
  :class:`ReplayHarness` samples a stored trace + resume point from the *live* buffer per rollout,
  so a growing buffer (this run's own rollouts) is picked up as it fills.

Replay kinds (``config.kinds``):

- ``recheck``          -> the full rollout + an appended "check your work" turn; re-roll.
- ``compaction_after`` -> resume from a compaction message; continue solving.
- ``compaction_before``-> resume before a compaction; the model writes the compaction, then continues.
- ``judge``            -> present the rollout's transcript and ask "was this correct?".

Scoring:

- ``recheck`` / ``compaction_*`` reuse the ORIGINAL env's verifier (``config.inner``): ``score``
  runs its rewards/metrics over the replay trace with the original task swapped in.
- ``judge`` compares the model's yes/no verdict against the original rollout's reward
  (``original_reward > judge_threshold``) — a self-supervised correctness label, no inner verifier.

The original task + reward ride on the replay task (offline) or in ``trace.info["replay"]`` (online).
"""

from __future__ import annotations

from typing import Literal

from pydantic import SerializeAsAny

from verifiers.v1.loaders import load_taskset
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, WireTask
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace

from replay.selector import (
    DEFAULT_FOLLOWUP,
    DEFAULT_KINDS,
    build_seed,
    iter_traces,
    resume_points,
    snapshot_ref_of,
)


class ReplayTask(Task):
    """A replayed seed plus the provenance needed to score it (offline mode; online mode carries
    the same provenance in ``trace.info["replay"]``)."""

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
    kinds: list[str] = DEFAULT_KINDS
    """Which replay kinds to source from (see module docstring); add ``"judge"`` to opt in."""
    followup: str = DEFAULT_FOLLOWUP
    """The user turn appended for ``recheck`` points."""
    judge_threshold: float = 0.5
    """``judge`` mode: the original rollout counts as correct when ``original_reward >`` this."""
    pool_size: int = 1024
    """Online mode: number of virtual task slots (num_tasks); the harness samples per rollout."""
    inner: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The ORIGINAL env's taskset (the verifier to reuse). Empty id => no inner scoring."""


def _parse_verdict(trace: Trace) -> bool | None:
    """The model's yes/no judgment from its last response (``judge`` mode). None if unclear."""
    msgs = trace.assistant_messages
    text = (msgs[-1].content or "").strip().lower() if msgs else ""
    if text.startswith("yes") or text.startswith("correct"):
        return True
    if text.startswith("no") or text.startswith("incorrect"):
        return False
    has_yes, has_no = "yes" in text, "no" in text
    return has_yes if has_yes != has_no else None


class ReplayTaskset(Taskset[ReplayTask, ReplayTasksetConfig]):
    def __init__(self, config: ReplayTasksetConfig) -> None:
        super().__init__(config)
        # Reuse the original env's verifier when configured (empty id => no inner scoring).
        self._inner: Taskset | None = load_taskset(config.inner) if config.inner.id else None

    def load_tasks(self) -> list[ReplayTask]:
        if self.config.mode == "online":
            # Virtual slots; ReplayHarness samples the live buffer per rollout.
            return [ReplayTask(idx=i, prompt=None) for i in range(self.config.pool_size)]
        kinds = set(self.config.kinds)
        tasks: list[ReplayTask] = []
        for src in iter_traces(self.config.buffer_glob):
            for pt in resume_points(src, kinds=kinds):
                tasks.append(
                    ReplayTask(
                        idx=len(tasks),
                        prompt=build_seed(src, pt, self.config.followup),
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
        # (Online restores inside ReplayHarness; judge mode needs no sandbox.) Skeleton:
        # snapshot capture isn't wired yet, so refs are None -> nothing to restore.
        if task.snapshot_ref is not None and task.resume_kind != "judge":
            await runtime.restore(task.snapshot_ref)

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        replay = trace.info.get("replay")  # online: harness-stashed provenance
        if replay:
            kind, original_dump, original_reward = (
                replay["kind"],
                replay["original_task"],
                replay["original_reward"],
            )
        else:
            t = trace.task
            kind = getattr(t, "resume_kind", "")
            original_dump = getattr(t, "original_task", {})
            original_reward = getattr(t, "original_reward", 0.0)

        if kind == "judge":
            # Grade the model's verdict against the original rollout's actual reward.
            verdict = _parse_verdict(trace)
            correct = original_reward > self.config.judge_threshold
            trace.record_reward("judge_match", 1.0 if verdict == correct else 0.0)
            return

        # recheck / compaction_*: reuse the ORIGINAL env's verifier over the replay trace,
        # with the original task swapped in.
        if self._inner is None or not original_dump:
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
