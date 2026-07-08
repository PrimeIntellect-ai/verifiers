"""The task: one problem to solve — its data AND its behavior.

A `Task` is a frozen pydantic model that carries everything a rollout needs: the
typed, task-specific data fields (the reference answer, ground truths, images, ...)
and the per-task behavior — runtime prep (`setup`/`finalize`), tools (`tools`), a
user simulator (`user`), well-formedness (`validate`), and judgement (`@reward`/
`@metric`/`@group_reward` methods, run by `score`/`score_group`). Subclass `Task`
per dataset; the `Taskset` is just the loader that yields these (see
`verifiers.v1.taskset`).

Because behavior lives on the task, a heterogeneous run — tasks from different
datasets, each with its own verification — is just a list of differently-typed
tasks; nothing above them branches on a type field.

It is the single judgement authority, scored at two granularities (execution lives
in the Rollout — per-rollout — and the Episode — group — which call these):
  - `score` runs `@reward`/`@metric` — plus the task's attached config-plugged
    judges (the `judges` field, from `TasksetConfig.judges`; see
    `verifiers.v1.judge`) — over one trace (in its live runtime).
  - `score_group` runs `@group_reward` over all the rollouts of this task at once —
    pairwise/preference rewards that compare samples.

Tasks are frozen and shared across their rollouts (a group's `n` samples hold the
same instance), so hook and scoring methods must not stash per-rollout state on
`self` — that lives on the trace (`trace.state`, typed via `Task[MyState]`).

Scoring after deserialization needs the real class: a `WireTask` carries the data
but none of the behavior, so a consumer that re-scores (e.g. `replay`) upgrades it
first. Every dump records its concrete class (`task_class`), resolved back within the
taskset's declared task type's subclass tree (`resolve_task_class`) — so a taskset
whose `load()` mixes task types round-trips each row as the right one.
"""

import asyncio
import inspect
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, ClassVar, Generic, TypeVar

from pydantic import (
    ConfigDict,
    Field,
    GetPydanticSchema,
    computed_field,
    model_validator,
)
from pydantic_core import core_schema

from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import TasksetError, boundary
from verifiers.v1.judge import Judge
from verifiers.v1.state import StateT
from verifiers.v1.types import Messages, StrictBaseModel, content_text

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset, User
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

# Method/property names on `Task` that a subclass must not redeclare as pydantic fields:
# the field value would shadow the bound method on the instance, so framework calls like
# `task.tools()` would call the data. Enforced at class definition
# (`Task.__pydantic_init_subclass__`).
_RESERVED_NAMES = frozenset(
    {
        "tools",
        "user",
        "setup",
        "finalize",
        "validate",
        "score",
        "score_group",
        "offline_skipped",
        "prompt_text",
        "task_class",
    }
)


def _class_path(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _requires_runtime(fn) -> bool:
    """A signal with a default-less `runtime` parameter can't run without a live runtime.
    A `runtime` parameter with a default (e.g. `runtime=None`) is optional — `invoke`
    just omits it — so those still run offline."""
    param = inspect.signature(fn).parameters.get("runtime")
    return param is not None and param.default is inspect.Parameter.empty


class TaskResources(StrictBaseModel):
    """Runtime resources a task requests (all optional), in Modal's units. Applied to the
    runtime config where the field exists; a field the runtime doesn't support is warned
    about and ignored. Precedence: cli/toml > task > the runtime default (`None` here =
    use the runtime/provider default)."""

    model_config = ConfigDict(frozen=True)

    cpu: float | None = None
    """CPU cores."""
    memory: float | None = None
    """Memory in GB."""
    gpu: str | None = None
    """GPU spec, e.g. "A100" or "A100:2" (type[:count])."""
    disk: float | None = None
    """Disk in GB (enforced by prime; advisory on docker/modal)."""


class TaskTimeout(StrictBaseModel):
    """Per-task wall-clock timeout overrides (seconds, all optional), one per rollout stage. Each
    merges with the eval's `timeout` (`TimeoutConfig`): cli/toml > this > default (no limit).
    Frozen, like `TaskResources`."""

    model_config = ConfigDict(frozen=True)

    setup: float | None = None
    """The task's `setup` hook."""
    harness: float | None = None
    """The harness run."""
    finalize: float | None = None
    """The task's `finalize` hook."""
    scoring: float | None = None
    """Verify + rewards/metrics."""


class Task(StrictBaseModel, Generic[StateT]):
    """A single problem to solve: typed data fields plus the per-task behavior (hooks,
    tools, `@reward`/`@metric` scoring). Subclass per dataset; parameterize the state
    type via `Task[MyState]` when tool/user servers or scoring share typed per-rollout
    state (defaults to the empty base `State`)."""

    model_config = ConfigDict(frozen=True)

    NEEDS_CONTAINER: ClassVar[bool] = False
    """Whether this task only runs in a container runtime (docker/prime). When True the
    Environment refuses the subprocess runtime — for tasks whose work only makes sense
    inside a per-task image (e.g. a SWE repo sandbox)."""

    idx: int
    """Stable integer index of this example within its taskset."""
    name: str | None = None
    """Optional human-readable task name/label (for display/filtering)."""
    description: str | None = None
    """Optional human-readable task description."""
    prompt: str | Messages | None
    """The user message shown to the model (the task's question/framing). Usually a `str`; a
    `Messages` list seeds a full initial conversation (e.g. a user message carrying images) and
    is only accepted by harnesses that set `SUPPORTS_MESSAGE_PROMPT`. Required — set it
    explicitly to `None` to mean the task carries no prompt: the task's user simulator
    (`Task.user`) then opens the conversation, its first `respond` supplying the initial user
    turn before the model is ever called."""
    system_prompt: str | None = None
    """Optional system prompt. Harnesses that set `APPENDS_SYSTEM_PROMPT` emit it as a real
    system message (or their own mechanism); others prepend it to `prompt` (with a
    warning). See `Harness.resolve_prompt`."""
    image: str | None = None
    """Container image this task needs (e.g. its harbor environment). When set, the
    runtime must be a container (docker/prime): the Environment injects it into the
    runtime config and refuses the subprocess runtime, which has no container."""
    workdir: str | None = None
    """Working directory the harness and scoring run in — the Environment injects it into
    the runtime config's `workdir` (where the runtime supports one). For a containerized
    task whose image puts the working tree at a non-default path (e.g. a SWE row's
    `/workspace/<repo>`)."""
    timeout: TaskTimeout = TaskTimeout()
    """Optional per-task timeout overrides, one per rollout stage (merge with the eval's `timeout`)."""
    resources: TaskResources = TaskResources()
    """Optional runtime resources this task requests (applied where supported)."""
    judges: Annotated[
        tuple[Judge, ...],
        # Judge is a plain class; validate/serialize as "any" without opening the whole
        # model to arbitrary types. Never serialized (excluded), so the wire is unchanged.
        GetPydanticSchema(lambda _source, _handler: core_schema.any_schema()),
    ] = Field(default=(), exclude=True, repr=False)
    """The config-plugged judges (built from `TasksetConfig.judges`), attached by the
    scoring caller — the `Environment` at episode time, `replay` before re-scoring.
    `score` runs them after the task's own `@reward`s. Excluded from serialization:
    judges are runtime objects, plugged back in from config on every run."""

    @computed_field(repr=False)  # type: ignore[prop-decorator]
    @property
    def task_class(self) -> str:
        """The concrete class this task serializes from (`module:qualname`), stamped into
        every dump — so an offline consumer (`replay`) can rebuild each row as its actual
        subclass when a taskset's `load()` mixes task types (see `resolve_task_class`)."""
        return _class_path(type(self))

    @model_validator(mode="before")
    @classmethod
    def _accept_own_dump(cls, data):
        # `task_class` is computed on output; a strict Task must still round-trip its own
        # dump, so drop the key on input. `WireTask` (extra="allow") keeps it instead —
        # in `model_extra` — to preserve the recorded class across wire round-trips.
        if isinstance(data, Mapping) and cls.model_config.get("extra") != "allow":
            data = {key: value for key, value in data.items() if key != "task_class"}
        return data

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        reserved = _RESERVED_NAMES & cls.model_fields.keys()
        if reserved:
            raise TypeError(
                f"{cls.__name__} declares field(s) {sorted(reserved)} that would shadow "
                f"the Task method(s) of the same name — rename the field(s) "
                f"(e.g. `user` -> `user_config`)"
            )

    @property
    def prompt_text(self) -> str:
        """The prompt as plain text — `prompt` itself when it's a `str`, the joined text
        content of a `Messages` prompt (images are dropped), `""` when the task carries no
        prompt. For consumers that need the task's framing as text regardless of the prompt
        form, e.g. the built-in judges' prompt templates."""
        if isinstance(self.prompt, str):
            return self.prompt
        texts = [content_text(message.content) for message in self.prompt or []]
        return "\n\n".join(text for text in texts if text)

    # ---- per-task behavior ---------------------------------------------------

    def tools(self) -> "list[Toolset]":
        """Tool servers exposing this task's tools to the model — `vf.Toolset`s (classes with
        `@vf.tool` methods), each carrying its `config` (placement / runtime; a remote `url`
        for an already-running server). Empty by default; override to give the task tools."""
        return []

    def user(self) -> "User | None":
        """A user simulator for this task — structurally a tool server (an MCP server
        with a runtime), but driven by the framework, not exposed to the model. After
        each model turn the interception server calls its `respond` tool and injects the
        reply as a user turn. None by default; override to make the task a simulated
        multi-turn conversation (e.g. a TextArena game)."""
        return None

    async def setup(self, trace: "Trace", runtime: "Runtime") -> None:
        """Prepare the live runtime for this task, after `runtime.start()` and before the
        harness runs. No-op by default; override to run per-task setup in the runtime (e.g.
        a SWE row checking out its base commit). Errors propagate and fail the rollout.

        Like the scoring hooks, `setup` declares the inputs it needs *by parameter name* and
        the framework injects them: any subset of `trace`, `runtime`. The trace (and its
        per-rollout `trace.state`) already exists when `setup` runs, so an override may
        stash per-rollout state there — e.g. `setup(self, trace, runtime)` or just
        `setup(self, runtime)` both work."""
        return None

    async def finalize(self, trace: "Trace", runtime: "Runtime") -> None:
        """Post-process the live runtime after the harness finishes, before scoring. No-op
        by default; override to do per-rollout work the rewards depend on — apply/commit the
        agent's diff, run a build, snapshot state, scrape runtime artifacts into `trace.info`.
        Runs while the runtime is still live (after generation, before `@reward`/`@metric`); the
        symmetric counterpart to `setup`. Declares its inputs by parameter name, like `setup`.
        Errors propagate and fail the rollout."""
        return None

    async def validate(self, runtime: "Runtime") -> bool:
        """Check the task is well-formed and solvable, independent of any model rollout — run
        by the `validate` entrypoint, never during a rollout. Valid (True) by default;
        override to assert the ground truth holds (e.g. a SWE row applying its gold patch and
        running its tests, or gsm8k confirming the verifier accepts the gold answer). Runs in
        a live runtime started for the task with `setup` already applied (a pure-data check
        can ignore it). Return False — or raise — to mark the task invalid; the entrypoint
        records the reason (the raised error's message)."""
        return True

    # ---- scoring ---------------------------------------------------------------

    async def score(
        self,
        trace: "Trace",
        runtime: "Runtime | None" = None,
    ) -> None:
        """Score one rollout: run all `@metric`, then `@reward`, then the attached
        config-plugged `judges` (see the `judges` field) over its trace, concurrently
        within each phase. Each metric is recorded in `trace.metrics` (a number, or a mapping
        merged in); each reward and judge verdict (weighted — likewise a number or a mapping
        merged in) in `trace.rewards`, which `trace.reward` sums. Signals declare what they
        need — `trace`, `runtime` (`self` is the task) — so a reward is either a pure function
        of the trace or runs read/write/exec in that (still-live) runtime, e.g. a verifier
        script.

        `runtime` may be `None` — the offline path taken by `replay`, which re-scores a saved
        trace with no live runtime. Signals that declare a `runtime` parameter are then skipped
        (they can't run without one) and logged; trace-only `@metric`/`@reward`s and the plugged
        judges (which grade from the trace) still run, overwriting their own keys and leaving the
        skipped signals' previously-recorded values untouched. With a runtime present (the eval
        path) nothing is skipped."""
        judges = self.judges
        available = {"task": self, "trace": trace}
        if runtime is not None:
            available["runtime"] = runtime

        def can_run(fn) -> bool:
            # A signal that *requires* a runtime can't run without one, so skip it when replaying.
            return runtime is not None or not _requires_runtime(fn)

        async with boundary(TasksetError, f"task {type(self).__name__} scoring"):
            metrics = [fn for fn in discover_decorated(self, "metric") if can_run(fn)]
            metric_results = (
                [await invoke(fn, available) for fn in metrics]
                if len(metrics) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in metrics))
            )
            for fn, result in zip(metrics, metric_results):
                if isinstance(result, Mapping):
                    trace.record_metrics(result)
                else:
                    trace.record_metric(fn.__name__, result)
            rewards = [fn for fn in discover_decorated(self, "reward") if can_run(fn)]
            reward_results = (
                [await invoke(fn, available) for fn in rewards]
                if len(rewards) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in rewards))
            )
            for fn, result in zip(rewards, reward_results):
                weight = getattr(fn, "_vf_weight", 1.0)
                if isinstance(result, Mapping):
                    for name, value in result.items():
                        trace.record_reward(name, value, weight)
                else:
                    trace.record_reward(fn.__name__, result, weight)
            runnable_judges = [judge for judge in judges if can_run(judge.score)]
            if runtime is None:
                # Exactly the signals `can_run` filtered out above — those requiring a runtime
                # (a `runtime` param with no default). Signals whose `runtime` has a default run
                # offline and are not reported as skipped.
                skipped = self.offline_skipped()
                if skipped:
                    logger.info(
                        "score: no runtime — skipped runtime-dependent signals: %s",
                        skipped,
                    )
            judge_results = (
                [await invoke(judge.score, available) for judge in runnable_judges]
                if len(runnable_judges) < 2
                else await asyncio.gather(
                    *(invoke(judge.score, available) for judge in runnable_judges)
                )
            )
            for judge, result in zip(runnable_judges, judge_results):
                if isinstance(result, Mapping):
                    for name, value in result.items():
                        trace.record_reward(name, value, judge.config.weight)
                else:
                    trace.record_reward(judge.reward_name, result, judge.config.weight)
            if runtime is not None and isinstance(trace.info, dict):
                # Map each runtime-requiring signal to the reward/metric keys it actually
                # produced — a Mapping result records under its own keys, not the method
                # name — so an offline re-score (`replay`) can restore exactly these
                # (`offline_skipped` names the signals; this maps them to their keys).
                def keys(result, fallback: str) -> list[str]:
                    return list(result) if isinstance(result, Mapping) else [fallback]

                produced = {
                    fn.__name__: keys(result, fn.__name__)
                    for fn, result in (
                        *zip(metrics, metric_results),
                        *zip(rewards, reward_results),
                    )
                    if _requires_runtime(fn)
                } | {
                    judge.reward_name: keys(result, judge.reward_name)
                    for judge, result in zip(runnable_judges, judge_results)
                    if _requires_runtime(judge.score)
                }
                if produced:
                    trace.info["offline_keys"] = produced

    def offline_skipped(self) -> list[str]:
        """The names of the signals `score` skips when re-scoring without a runtime (they
        declare a default-less `runtime` parameter): this task's `@metric`/`@reward`s plus
        its attached judges. Lets an offline consumer (`replay`) preserve their previously
        recorded values instead of dropping them — expanded to the keys each signal
        actually recorded via the trace's `info["offline_keys"]` (a Mapping-returning
        signal records under its own keys, not its name)."""
        return [
            fn.__name__
            for fn in (
                *discover_decorated(self, "metric"),
                *discover_decorated(self, "reward"),
            )
            if _requires_runtime(fn)
        ] + [
            judge.reward_name for judge in self.judges if _requires_runtime(judge.score)
        ]

    async def score_group(self, traces: "list[Trace]") -> None:
        """Score a group of rollouts of this task: run every `@group_reward` over all
        the traces at once (pairwise/preference rewards), each returning one score per
        trace, aligned to `traces`. A group reward declares what it needs — `traces`
        (`self` is the shared task) — and compares trace metadata (anything from the
        runtime is recorded per rollout as a `@metric` first). Scores are weighted into
        each trace's reward, alongside the per-rollout rewards. No-op without `@group_reward`s."""
        rewards = discover_decorated(self, "group_reward")
        if not rewards:
            return
        available = {"task": self, "traces": traces}
        async with boundary(TasksetError, f"task {type(self).__name__} group scoring"):
            reward_results = (
                [await invoke(fn, available) for fn in rewards]
                if len(rewards) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in rewards))
            )
            for fn, scores in zip(rewards, reward_results):
                if len(scores) != len(traces):
                    raise ValueError(
                        f"@group_reward {fn.__name__} returned {len(scores)} score(s) "
                        f"for {len(traces)} rollout(s); it must return one per trace"
                    )
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace, score in zip(traces, scores):
                    trace.record_reward(fn.__name__, score, weight)


class WireTask(Task):
    """A `Task` that accepts (and preserves) taskset-specific extra fields. Lets a `Trace`
    be typed on the wire — `Trace[WireTask]` — without importing the taskset, since the real
    `Task` subclass's extra fields land in `model_extra` instead of being rejected. It carries
    the data but none of the subclass's behavior (no hooks, no `@reward`s) — a caller that
    re-scores upgrades to the real type via `task_type(taskset_id)` first."""

    model_config = ConfigDict(extra="allow")

    @computed_field(repr=False)  # type: ignore[prop-decorator]
    @property
    def task_class(self) -> str:
        # A WireTask is a container for some other class's row: re-serializing must keep
        # the recorded class, not stamp its own.
        recorded = (self.model_extra or {}).get("task_class")
        return recorded if isinstance(recorded, str) else _class_path(type(self))


TaskT = TypeVar("TaskT", bound=Task)


def resolve_task_class(base: type[TaskT], path: str | None) -> type[TaskT]:
    """The concrete Task class a wire row recorded as `path` (`module:qualname`, see
    `Task.task_class`), resolved strictly within `base`'s subclass tree — only classes the
    taskset plugin already imported, never an arbitrary import. Falls back to `base` when
    the row predates the field or records a class that no longer exists."""

    def walk(cls: type[TaskT]):
        yield cls
        for sub in cls.__subclasses__():
            yield from walk(sub)

    if path:
        for cls in walk(base):
            if _class_path(cls) == path:
                return cls
    return base
