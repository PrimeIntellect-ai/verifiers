"""The replay taskset base: turn saved rollouts back into training tasks.

Reads a run's saved rollout files (``<output>/rollouts/step_*/train_rollouts.jsonl``,
one ``Trace.to_record()`` line each — the prime-rl orchestrator's rollout output) as a
buffer and serves tasks derived from them. Like ``harbor`` or ``textarena``, this is a
base for thin subclass packages: a derivation subclasses `ReplayTaskset` (binding its
narrowed config in the generic, so the loader picks it up) and implements two hooks —

- ``record_anchors(nodes, children, roots, tree)``: the resume points one saved rollout
  offers (each becomes one task; ``None`` anchors at the rollout's final state).
- ``build_prompt(record, anchor)``: the seeded conversation for one anchor.

Everything else is shared: the lazy buffer (tasks bind per request from (file, offset)
handles — saved lines average ~1MB), online semantics over a growing run dir, scoring/
tools/setup/finalize delegation to the original (``inner``) taskset, and lineage so
replay-derived records aren't re-replayed by default. Run replay tasksets under the
source env's harness (it must support message prompts).
"""

import asyncio
import logging

from pydantic import PrivateAttr, SerializeAsAny, model_validator

from verifiers.v1.decorators import discover_decorated, metric
from verifiers.v1.loaders import narrow_plugin_field, task_type, taskset_class, taskset_config_type
from verifiers.v1.mcp import Toolset
from verifiers.v1.runtimes import Runtime
from verifiers.v1.state import State, state_cls
from verifiers.v1.task import Task
from verifiers.v1.taskset import ConfigT, Taskset, TasksetConfig
from verifiers.v1.tasksets.replay.buffer import ReplayBuffer
from verifiers.v1.tasksets.replay.surgery import build_children, main_tree, unwrap_source_task
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class ReplayTasksetConfig(TasksetConfig):
    """Shared config of every replay derivation: where the buffer is, whose rollouts to
    replay, and the original taskset that scores the derived rollouts."""

    buffer_dir: str = ""
    """The saved-rollout dir to replay: a run's `rollouts` dir (or the run dir containing
    it). Under prime-rl the literal `"self"` resolves to this run's own rollout dir (an
    online buffer over the run's freshly written rollouts)."""

    inner: SerializeAsAny[TasksetConfig]
    """The original taskset's config — it scores the derived rollouts and provides
    their tools, so it must reproduce the source run's taskset config."""

    source_envs: list[str] | None = None
    """Which envs' rollouts to replay, by their stamped name (`info.prime_rl.env_name`).
    None (the default) replays every env except replay envs — deriving from a replay
    env's own outputs is a feedback loop unless chosen deliberately. Listing env names
    replays exactly those, and naming a replay env is that deliberate choice: chained
    derivations (recheck a recheck) are expressed as one replay env sourcing another.
    With an explicit list, records without the stamp never match."""

    allow_container: bool = False
    """Allow sources whose task ran in a container image. The container state the
    transcript references is gone — a fresh container is provisioned from the same
    image, so the model resumes in a reset world. Off until that's fair (sandbox
    snapshotting)."""

    online: bool = False
    """Treat the buffer as growing: rescan for new steps during training, read only
    barrier-complete steps, and sample a fresh source per request (which forces
    whole-group dispatch so GRPO groups share one source). Set automatically for
    `buffer_dir = "self"`; set it yourself only to watch another still-running run's
    dir. Offline buffers are indexed once, deterministically per task index."""

    @model_validator(mode="before")
    @classmethod
    def _narrow_inner(cls, data):
        if isinstance(data, dict) and data.get("inner"):
            narrow_plugin_field(data, "inner", taskset_config_type)
        return data

    @model_validator(mode="after")
    def _resolve(self):
        if not self.buffer_dir:
            raise ValueError('replay tasksets need `buffer_dir` (a run\'s rollouts dir, or "self" under prime-rl)')
        if self.buffer_dir == "self":
            # prime-rl rewrites the sentinel to a resolved path before the env server
            # spawns, so onlineness must be pinned while the sentinel is visible.
            self.online = True
        return self


class ReplayTask(Task):
    """A derived task plus the lazy handle onto its source rollout. Stubs (from
    `load_tasks`) carry `prompt=None` and an empty `source_task`; `resolve_task`
    materializes the real thing. `source_task`/`source_id` double as the lineage
    marker replay buffers use to skip replay-derived records by default."""

    kind: str = ""
    """The derivation that minted this task (its taskset id)."""
    source_task: dict = {}
    """The source rollout's saved task dict, verbatim — rebuilt into the original typed
    task for scoring/tool delegation."""
    original_reward: float = 0.0
    """The reward the source rollout actually received."""
    source_id: str = ""
    source_step: int = -1

    # The rebuilt typed original task, cached at materialization so the per-rollout
    # hooks (tools/setup/finalize/score) don't re-validate the dict. Private attrs
    # bypass frozen and never serialize.
    _inner: Task | None = PrivateAttr(default=None)


class ReplayTaskset(Taskset[ReplayTask, ConfigT, State]):
    """Base for replay derivations; subclass with a narrowed config bound in the
    generic (`class MyTaskset(ReplayTaskset, Taskset[ReplayTask, MyConfig])`) and
    implement `record_anchors` + `build_prompt`."""

    def __init__(self, config: ConfigT) -> None:
        super().__init__(config)
        inner_config = config.inner
        self.inner: Taskset = taskset_class(inner_config.id)(inner_config)
        if discover_decorated(self.inner, "group_reward"):
            raise ValueError(
                f"taskset {inner_config.id!r} defines @group_reward(s); replay cannot delegate group "
                "scoring (the group would score against the replay task, not the original)"
            )
        if state_cls(type(self.inner)) is not State:
            raise ValueError(
                f"taskset {inner_config.id!r} uses a custom State; replay rollouts build the base "
                "State, so its typed state would never be populated"
            )
        if type(self.inner).user is not Taskset.user:
            raise ValueError(f"taskset {inner_config.id!r} defines a user simulator; replay does not support one")
        self.inner_task_type = task_type(inner_config.id)
        self.NEEDS_CONTAINER = self.inner.NEEDS_CONTAINER
        # Online buffers sample a fresh source per request; the whole GRPO group must
        # share that draw, so all its rollouts must arrive as one run_group request.
        self.REQUIRES_GROUP_ROLLOUTS = config.online
        self.buffer = ReplayBuffer(
            buffer_dir=config.buffer_dir,
            anchors=self._anchors,
            online=config.online,
            source_envs=config.source_envs,
            allow_container=config.allow_container,
        )

    # ------------------------------------------------------------- derivation hooks

    def record_anchors(
        self, record: dict, children: dict[int, list[int]], roots: list[int], tree: set[int]
    ) -> list[int | None]:
        """The resume points one saved rollout offers this derivation — each becomes one
        task. Return node indices (see `surgery` for enumerators), `[None]` to anchor at
        the rollout's final state, or `[]` when the rollout offers nothing."""
        raise NotImplementedError

    def build_prompt(self, record: dict, anchor: int | None) -> str | list[dict]:
        """The seeded conversation for one (source record, anchor) pair."""
        raise NotImplementedError

    def _anchors(self, record: dict) -> list[int | None]:
        children, roots = build_children(record["nodes"])
        return self.record_anchors(record, children, roots, main_tree(children))

    # ------------------------------------------------------------------ tasks

    def load_tasks(self) -> list[ReplayTask]:
        if self.config.buffer_dir == "self":
            raise ValueError(
                'buffer_dir = "self" is resolved by the prime-rl orchestrator before env servers '
                "spawn; when serving this taskset standalone, pass an explicit rollouts dir"
            )
        candidates = self.buffer.scan()
        if self.config.online:
            # Sampling ignores the task index, so an online buffer serves exactly one
            # virtual task. (prime-rl's no-ratio fallback weights envs by task count —
            # replay envs need an explicit ratio regardless.)
            num_tasks = 1
        else:
            if not candidates:
                raise ValueError(f"replay buffer at {self.buffer.rollout_dir} has no replayable candidates")
            num_tasks = len(candidates)
        return [ReplayTask(idx=i, kind=self.config.id, prompt=None) for i in range(num_tasks)]

    async def resolve_task(self, task: ReplayTask) -> ReplayTask:
        # The read (~1MB line), graph walk, and prompt build all run off the event loop.
        if not self.config.online:
            candidate = self.buffer.pick(task.idx)
            return await asyncio.to_thread(self._materialize, task.idx, candidate)
        # An online source line can vanish under us (its run resumed and cleaned future
        # steps): drop the dangling candidate and draw again instead of failing forever.
        for _ in range(8):
            candidate = await self.buffer.sample()
            try:
                return await asyncio.to_thread(self._materialize, task.idx, candidate)
            except FileNotFoundError:
                logger.warning("replay source %s vanished; discarding candidate", candidate.path)
                self.buffer.discard(candidate)
        raise RuntimeError(f"replay buffer at {self.buffer.rollout_dir} keeps serving vanished source files")

    def _materialize(self, idx: int, candidate) -> ReplayTask:
        record = self.buffer.read_record(candidate)
        prompt = self.build_prompt(record, candidate.anchor_node)
        # A chained source (a replay env sourcing another replay env) nests its lineage;
        # scoring, tools, and provisioning are always keyed on the innermost original.
        source_task = unwrap_source_task(record["task"])
        provision = {
            key: value for key in ("image", "workdir", "timeout", "resources") if (value := source_task.get(key))
        }
        task = ReplayTask(
            idx=idx,
            name=f"{self.config.id}:{candidate.source_id[:8]}",
            prompt=prompt,
            kind=self.config.id,
            source_task=source_task,
            original_reward=candidate.original_reward,
            source_id=candidate.source_id,
            source_step=candidate.step,
            **provision,
        )
        # Rebuild the typed original task now — inner-taskset schema drift fails loudly
        # here, before any generation is spent — and cache it for the per-rollout hooks.
        task._inner = self.inner_task_type.model_validate(source_task)
        return task

    def _inner_task(self, task: ReplayTask) -> Task:
        if task._inner is None:
            task._inner = self.inner_task_type.model_validate(task.source_task)
        return task._inner

    # ---------------------------------------------------- original-world hooks

    def tools(self, task: ReplayTask) -> list[Toolset]:
        # Stubs (env serving inspects tools(tasks[0]) for shared placement) get none;
        # inner tasksets with shared-placement toolsets are therefore unsupported.
        if not task.source_task:
            return []
        return self.inner.tools(self._inner_task(task))

    async def setup(self, task: ReplayTask, runtime: Runtime) -> None:
        await self.inner.setup(self._inner_task(task), runtime)

    async def finalize(self, task: ReplayTask, trace: Trace, runtime: Runtime) -> None:
        inner_task = self._inner_task(task)
        # The shallow copy shares nodes/info/state with the real trace, so anything the
        # inner taskset scrapes for its rewards lands where scoring will read it.
        await self.inner.finalize(inner_task, trace.model_copy(update={"task": inner_task}), runtime)

    # ------------------------------------------------------------------ scoring

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        # Score with the original taskset, seen through a view whose `task` is the
        # rebuilt original: the shallow copy shares the rewards/metrics/info dicts,
        # so the inner rewards land on the real trace with their original names and
        # weights, while `trace.task` stays the ReplayTask for everything else.
        inner_task = self._inner_task(trace.task)
        await self.inner.score(trace.model_copy(update={"task": inner_task}), runtime)
        await super().score(trace, runtime)

    @metric
    async def replay_stats(self, task: ReplayTask, trace: Trace) -> dict[str, float]:
        return {"replay/source_reward": task.original_reward, "replay/source_step": float(task.source_step)}
