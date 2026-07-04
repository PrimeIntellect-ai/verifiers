"""replay — re-enter saved rollouts as fresh tasks: continue them mid-way, or recheck them.

Loads rollout records (``Trace.to_record()`` JSONL lines, as prime-rl writes to
``<output_dir>/rollouts/step_*/*_rollouts.jsonl``) and turns each into a task that re-enters the
original task from a resume point. The ``source`` taskset — the one the records came from — is
loaded by id and provides everything but the seed: tools, user simulator, setup/finalize, and
scoring, so a new completion is judged by the original env's own verifier. Each replayed task
*is* the original typed task with its ``prompt`` swapped for the seed, so task-specific fields
(reference answers, images, timeouts) flow into lifecycle hooks and rewards unchanged.

Two built-in modes:

- ``continue`` — resume a rollout and finish the job. ``anchor = "compaction"`` re-enters at
  each context restart, detected structurally from the trace's branch graph — no
  compaction-prompt matching, so records from different harnesses (each with its own handoff
  wording and shape) all qualify. The canonical ``[system, user(summary)]`` restart seeds as a
  plain string prompt and works under every harness, including agent CLIs like ``rlm``.
  ``anchor = "tool-call"`` re-enters right after one deterministically-drawn tool result: the
  seed is the conversation up to that point as a ``Messages`` prompt, which needs a
  message-seeding harness (``default``/``null``).
- ``recheck`` — hand the model its finished attempt (the final branch, truncation artifacts
  stripped) followed by a new user turn asking it to verify and fix its work. Also a
  ``Messages`` prompt. ``last_reply``-style rewards score the revision: seeded context is
  never ``sampled``, so it is invisible to them.

Other recycling schemes are a subclass away: :meth:`ReplayTaskset.seeds` is the override
point — a taskset package subclassing ``ReplayTaskset`` inherits the record loading, source
delegation, and snapshot plumbing, and only redefines how a source trace becomes seeds.

The runtime is fresh: ``setup`` runs anew. When the source run captured sandbox snapshots
(``trace.info["snapshots"]``, see ``records.SNAPSHOTS_INFO_KEY``), the seed's anchor snapshot
is restored after ``setup`` — the plug-in point is :func:`restore_snapshot`, which fails
loudly until runtimes support it. Without snapshots, mid-trajectory seeds suit tasks whose
state the model can rebuild (or that verify pure text). Run with the same harness config as
the source run so the rebuilt system prompt and tool inventory match the seeded context.
"""

import inspect
import logging
import random
from functools import cached_property
from glob import glob
from pathlib import Path
from typing import Literal

from pydantic import SerializeAsAny, ValidationError, model_validator

from verifiers.v1.loaders import (
    load_taskset,
    narrow_plugin_field,
    task_type,
    taskset_config_type,
)
from verifiers.v1.mcp import Toolset, User
from verifiers.v1.runtimes import Runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.tasksets.replay.records import (
    RECHECK_PROMPT,
    Seed,
    compaction_seeds,
    iter_records,
    recheck_seed,
    tool_call_seed,
)
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


async def restore_snapshot(runtime: Runtime, ref: str) -> None:
    """Restore a source rollout's sandbox snapshot into the fresh runtime, so a mid-trajectory
    resume sees the filesystem/session state its seeded context claims exists. Runtimes can't
    capture or restore snapshots yet — no producer records refs today, so this only fires once
    one does; plug the runtime restore call in here when support lands."""
    raise NotImplementedError(
        f"sandbox snapshot restore is not supported by runtimes yet (ref {ref!r})"
    )


class ReplayConfig(TasksetConfig):
    """The replay taskset: which records to re-enter, how, and whose env they came from."""

    records: str | list[str] = []
    """Glob pattern(s) of rollout record files — ``Trace.to_record()`` JSONL lines, e.g.
    ``"<output_dir>/rollouts/step_*/train_rollouts.jsonl"``. Required. Validation freezes the
    globs to the concrete matched files, so every env-server pool worker builds the identical
    task list even if the directory keeps growing — point it at a finished run's records, not
    a live producer's. Lines that don't validate as the source taskset's task type (e.g.
    another env's rollouts in a mixed train file) are skipped and counted; that filter is
    structural, so a source whose task type has no distinguishing required fields can't tell
    foreign records apart — use single-env record files for such sources."""

    source: SerializeAsAny[TasksetConfig] = TasksetConfig()
    """The taskset the records came from (``id`` required, plus its own fields), resolved to
    its concrete config type by id. It supplies tools, user simulator, setup/finalize, and
    scoring for every replayed task."""

    mode: Literal["continue", "recheck"] = "continue"
    """``continue`` resumes a rollout from a resume point (see ``anchor``); ``recheck`` replays
    the finished attempt and appends a user turn asking the model to verify and fix it."""

    anchor: Literal["compaction", "tool-call"] = "compaction"
    """Where ``continue`` re-enters: ``compaction`` seeds each recorded post-compaction restart
    prompt (any harness; only rollouts that compacted become sources); ``tool-call`` seeds the
    conversation through one drawn tool result (needs a message-seeding harness)."""

    recheck_prompt: str = RECHECK_PROMPT
    """The verification request appended as the new user turn in ``recheck`` mode."""

    max_seed_tokens: int | None = None
    """Skip seeds whose context exceeds this many tokens (recorded count when the record
    carries token ids, else a chars/4 estimate). Unset admits every seed — but a seed near the
    trainer's ``seq_len`` leaves no room to sample, so set this for long-context sources."""

    seed: int = 0
    """Salt for the per-record draw of ``tool-call`` resume points (each record's draw is
    deterministic in this and the source trace id)."""

    @model_validator(mode="before")
    @classmethod
    def _narrow_source(cls, data):
        if isinstance(data, dict):
            narrow_plugin_field(data, "source", taskset_config_type)
        return data

    @model_validator(mode="after")
    def _validate(self):
        if not self.records:
            raise ValueError(
                "replay requires `records` — glob pattern(s) of rollout JSONL files"
            )
        if not self.source.id:
            raise ValueError(
                "replay requires `source.id` — the taskset the records came from"
            )
        # Compare against the default rather than `model_fields_set`: a dumped resolved config
        # writes every field explicitly, and must re-validate cleanly.
        if self.mode == "recheck" and self.anchor != "compaction":
            raise ValueError("`anchor` only applies to mode='continue'")
        # Freeze globs to the matched files: every env-server pool worker re-validates this
        # config and rebuilds the task list independently, so the file set must not drift as
        # the records directory grows (an elastic pool spawns workers throughout the run).
        # No matches leaves the patterns untouched for `load_tasks` to fail on — the files may
        # not exist where the config is first validated (e.g. a dry run on another host).
        patterns = [self.records] if isinstance(self.records, str) else self.records
        matched = sorted(
            {path for pattern in patterns for path in glob(pattern, recursive=True)}
        )
        if matched:
            self.records = matched
        return self


class ReplayTaskset(Taskset[Task, ReplayConfig]):
    _snapshots: dict[int, str] = {}
    """Task idx -> sandbox snapshot ref of its seed's anchor node; rebuilt by `load_tasks`
    (the class default only backs a taskset used without loading, e.g. in tests)."""

    @cached_property
    def source(self) -> Taskset:
        """The source taskset, loaded by id. Every hook below delegates to it, so a replayed
        task runs with the original env's toolkit and is scored by its verifier."""
        return load_taskset(self.config.source)

    @property
    def NEEDS_CONTAINER(self) -> bool:  # type: ignore[override] — mirrors the source's ClassVar
        return type(self.source).NEEDS_CONTAINER

    @property
    def defines_tools(self) -> bool:
        # Report the source's capability, not this class's delegating stubs — the Environment's
        # harness gate must see exactly what the source taskset exposes.
        return self.source.defines_tools

    @property
    def defines_user(self) -> bool:
        return self.source.defines_user

    def state_type(self) -> type[State]:
        return self.source.state_type()

    def load_tasks(self) -> list[Task]:
        # Group rewards and stop conditions are found by inspecting the taskset instance
        # (the server's `requires_group_scoring`, the rollout's stop checks), so delegation
        # alone would hide the source's — surface its tagged methods here. Execution still
        # goes through the delegating `score_group`, so nothing runs twice.
        for name, method in inspect.getmembers(self.source, inspect.ismethod):
            if hasattr(method, "group_reward") or hasattr(method, "stop"):
                if hasattr(type(self), name):
                    raise ValueError(
                        f"source hook {self.config.source.id}.{name} shadows a replay taskset attribute"
                    )
                setattr(self, name, method)

        patterns = (
            [self.config.records]
            if isinstance(self.config.records, str)
            else self.config.records
        )
        paths = sorted(
            {path for pattern in patterns for path in glob(pattern, recursive=True)}
        )
        if not paths:
            raise ValueError(f"replay `records` matched no files: {patterns}")
        trace_cls = Trace[task_type(self.config.source.id)]

        tasks: list[Task] = []
        self._snapshots: dict[int, str] = {}
        skipped = {"foreign": 0, "empty": 0, "no_seed": 0, "overlong": 0}
        for record in iter_records(Path(path) for path in paths):
            try:
                trace = trace_cls.model_validate(record)
            except ValidationError:
                skipped["foreign"] += 1
                continue
            if not trace.nodes:
                skipped["empty"] += 1
                continue
            seeds = self.seeds(trace)
            if not seeds:
                skipped["no_seed"] += 1
                continue
            for seed in seeds:
                if (
                    self.config.max_seed_tokens is not None
                    and seed.tokens > self.config.max_seed_tokens
                ):
                    skipped["overlong"] += 1
                    continue
                if seed.snapshot is not None:
                    self._snapshots[len(tasks)] = seed.snapshot
                tasks.append(
                    trace.task.model_copy(
                        update={
                            "idx": len(tasks),
                            "name": seed.name,
                            "prompt": seed.prompt,
                        }
                    )
                )
        logger.info(
            "replay(%s/%s): %d tasks from %d files (skipped: %s)",
            self.config.mode,
            self.config.anchor if self.config.mode == "continue" else "-",
            len(tasks),
            len(paths),
            ", ".join(
                f"{reason}={count}" for reason, count in skipped.items() if count
            ),
        )
        if not tasks:
            raise ValueError(
                f"replay produced no tasks from {len(paths)} record files "
                f"(skipped: {skipped}) — check `records`, `source.id`, and `mode`/`anchor`"
            )
        return tasks

    def seeds(self, trace: Trace) -> list[Seed]:
        """Every seed a source trace yields under the configured mode. The override point for
        new recycling schemes: a taskset package subclassing ``ReplayTaskset`` redefines only
        this — record loading, source delegation, and snapshot restore stay inherited."""
        if self.config.mode == "recheck":
            seed = recheck_seed(trace, self.config.recheck_prompt)
            return [seed] if seed else []
        if self.config.anchor == "compaction":
            return compaction_seeds(trace)
        seed = tool_call_seed(trace, random.Random(f"{self.config.seed}:{trace.id}"))
        return [seed] if seed else []

    def tools(self, task: Task) -> list[Toolset]:
        return self.source.tools(task)

    def user(self, task: Task) -> User | None:
        return self.source.user(task)

    async def setup(self, task: Task, runtime: Runtime) -> None:
        await self.source.setup(task, runtime)
        ref = self._snapshots.get(task.idx)
        if ref is not None:
            await restore_snapshot(runtime, ref)

    async def finalize(self, task: Task, trace: Trace, runtime: Runtime) -> None:
        await self.source.finalize(task, trace, runtime)

    async def validate(self, task: Task, runtime: Runtime) -> bool:
        return await self.source.validate(task, runtime)

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        await self.source.score(trace, runtime)

    async def score_group(self, traces: list[Trace]) -> None:
        await self.source.score_group(traces)
