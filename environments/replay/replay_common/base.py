"""Shared base for the replay-buffer tasksets.

Each concrete taskset/harness fixes one ``KIND`` (``recheck`` / ``judge`` / ``compaction_after``
/ ``compaction_before``) and lives in its own selectable module (``replay_recheck``, ...). This
base holds everything they share: buffer sourcing (offline materialize + online sampling),
seed building, snapshot restore, and scoring.

Sourcing (``config.mode``):
- ``offline`` — ``load_tasks`` materializes one task per resume point of this taskset's ``KIND``;
  the bundled harness sees a non-empty ``task.prompt`` and just runs the default chat loop.
- ``online`` — ``load_tasks`` returns ``pool_size`` virtual slots and the harness samples a
  ``KIND`` resume point from the *live* buffer per rollout.

Scoring:
- ``recheck`` / ``compaction_*`` reuse the ORIGINAL env's verifier (``config.inner``).
- ``judge`` grades the model's verdict against the original rollout's reward (no inner verifier).
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import SerializeAsAny

from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.default.harness import (
    PROGRAM_SOURCE,
    DefaultHarness,
    DefaultHarnessConfig,
)
from verifiers.v1.loaders import load_taskset
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import Task, WireTask
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace, WireTrace

from replay_common.selector import (
    DEFAULT_FOLLOWUP,
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
    snapshot_ref: str | None = None
    original_task: dict = {}
    original_reward: float = 0.0


class ReplayConfig(TasksetConfig):
    mode: Literal["offline", "online"] = "offline"
    buffer_glob: str = ""
    """Glob of stored-rollout JSONL files, e.g. ``.../rollouts/step_*/train_rollouts.jsonl``."""
    followup: str = DEFAULT_FOLLOWUP
    """The user turn appended for the ``recheck`` taskset."""
    judge_threshold: float = 0.5
    """``judge`` taskset: the original rollout counts as correct when ``original_reward >`` this."""
    pool_size: int = 1024
    """Online mode: number of virtual task slots (num_tasks); the harness samples per rollout."""
    inner: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The ORIGINAL env's taskset (the verifier to reuse). Empty id => no inner scoring."""


class ReplayHarnessConfig(DefaultHarnessConfig):
    buffer_glob: str = ""
    """Online mode: the live buffer to sample from."""
    followup: str = DEFAULT_FOLLOWUP


def _parse_verdict(trace: Trace) -> bool | None:
    """The model's yes/no judgment from its last response (``judge``). None if unclear."""
    msgs = trace.assistant_messages
    text = (msgs[-1].content or "").strip().lower() if msgs else ""
    if text.startswith("yes") or text.startswith("correct"):
        return True
    if text.startswith("no") or text.startswith("incorrect"):
        return False
    has_yes, has_no = "yes" in text, "no" in text
    return has_yes if has_yes != has_no else None


class BaseReplayTaskset(Taskset[ReplayTask, ReplayConfig]):
    """Concrete subclasses set ``KIND``; everything else is shared."""

    KIND: ClassVar[str] = ""

    def __init__(self, config: ReplayConfig) -> None:
        super().__init__(config)
        # Reuse the original env's verifier when configured (empty id => no inner scoring).
        self._inner: Taskset | None = load_taskset(config.inner) if config.inner.id else None

    def load_tasks(self) -> list[ReplayTask]:
        if self.config.mode == "online":
            return [ReplayTask(idx=i, prompt=None) for i in range(self.config.pool_size)]
        tasks: list[ReplayTask] = []
        for src in iter_traces(self.config.buffer_glob):
            for pt in resume_points(src, kinds={self.KIND}):
                tasks.append(
                    ReplayTask(
                        idx=len(tasks),
                        prompt=build_seed(src, pt, self.config.followup),
                        source_trace_id=src.id,
                        resume_node=pt["node"],
                        snapshot_ref=snapshot_ref_of(src, pt["node"]),
                        original_task=src.task.model_dump(),
                        original_reward=src.reward,
                    )
                )
        return tasks

    async def setup(self, task: ReplayTask, runtime: Runtime) -> None:
        # Offline exec/sandbox replay: restore to the resume point before the harness runs.
        # (Online restores inside the harness; judge needs no sandbox.) Skeleton: refs are None.
        if task.snapshot_ref is not None and self.KIND != "judge":
            await runtime.restore(task.snapshot_ref)

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        replay = trace.info.get("replay")  # online: harness-stashed provenance
        original_dump = replay["original_task"] if replay else getattr(trace.task, "original_task", {})
        original_reward = replay["original_reward"] if replay else getattr(trace.task, "original_reward", 0.0)

        if self.KIND == "judge":
            verdict = _parse_verdict(trace)
            correct = original_reward > self.config.judge_threshold
            trace.record_reward("judge_match", 1.0 if verdict == correct else 0.0)
            return

        # recheck / compaction_*: reuse the ORIGINAL verifier with the original task swapped in.
        if self._inner is None or not original_dump:
            return
        replay_task = trace.task
        trace.task = WireTask.model_validate(original_dump)
        try:
            await self._inner.score(trace, runtime)
        finally:
            trace.task = replay_task


class BaseReplayHarness(DefaultHarness):
    """Bundled with each taskset (auto-selected via ``default_harness_id``). Offline: a materialized
    ``task.prompt`` is present, so defer to the default chat loop. Online: sample a ``KIND`` resume
    point from the live buffer and seed the loop with it."""

    KIND: ClassVar[str] = ""
    SUPPORTS_MESSAGE_PROMPT = True

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        if trace.task.prompt is not None:  # offline: materialized seed -> default behavior
            return await super().launch(ctx, trace, runtime, endpoint, secret, mcp_urls)

        rng = random.Random(trace.id)
        sample = self._sample(rng)
        if sample is None:  # buffer empty (warmup) / no matching points yet
            trace.stop("replay_buffer_empty")
            return ProgramResult(exit_code=0, stdout="", stderr="")
        src, point = sample

        ref = snapshot_ref_of(src, point["node"])
        if ref is not None and self.KIND != "judge":  # judge needs no sandbox; skeleton refs are None
            await runtime.restore(ref)

        trace.info["replay"] = {
            "source_id": src.id,
            "resume_node": point["node"],
            "kind": point["kind"],
            "original_task": src.task.model_dump(),
            "original_reward": src.reward,
        }
        seed = build_seed(src, point, self.config.followup)
        env = {**self.config.env}
        env["INITIAL_MESSAGES"] = json.dumps([message_to_wire(m) for m in seed])
        args = [f"--base-url={endpoint}", f"--api-key={secret}", f"--model={ctx.model}"]
        if mcp_urls:
            args.append(
                "--mcp-config="
                + json.dumps({"mcpServers": {n: {"url": u} for n, u in mcp_urls.items()}})
            )
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)
        return await runtime.run_program([*program, *args], env)

    def _sample(self, rng: random.Random) -> tuple[Trace, dict] | None:
        """Scan the live buffer in random order; return the first (trace, ``KIND`` point) found."""
        files = sorted(glob.glob(self.config.buffer_glob))
        rng.shuffle(files)
        for path in files:
            try:
                lines = Path(path).read_text().splitlines()
            except OSError:
                continue
            rng.shuffle(lines)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                src = WireTrace.model_validate(json.loads(line))
                points = resume_points(src, kinds={self.KIND})
                if points:
                    return src, rng.choice(points)
        return None
