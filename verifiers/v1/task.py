"""The task: the unit of work — its data as frozen, typed fields; its behavior on the class.

A `Task` is what an agent consumes to produce a trace. Its *instance* is pure, serializable
data (the prompt, the ground truth, runtime requests) that rides the wire and persists with
the trace; its *class* carries everything an episode needs from the inside — lifecycle hooks
(`setup`/`finalize`/`validate`), tools (`load_tools`), a user simulator (`load_user`), and judgement
(`@vf.reward`/`@vf.metric`/`@vf.stop`/`@vf.group_reward` methods). Methods aren't fields, so
the split costs nothing: data serializes, behavior stays importable code.

Because behavior travels with the class, every way of minting a task is equal: a `Taskset`
factory deriving them from a dataset, a topology's `load_tasks`, a task constructed mid-`go`
from an upstream trace, a future replay buffer. A generated task is a first-class citizen —
question and verifier in one typed object.
"""

import asyncio
import inspect
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, TypeVar

from pydantic import ConfigDict

from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import TaskError, boundary
from verifiers.v1.state import State
from verifiers.v1.types import Messages, StrictBaseModel, content_text

if TYPE_CHECKING:
    from verifiers.v1.mcp import Toolset, User
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace


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


class Task(StrictBaseModel):
    """A single problem to solve. Subclass to add typed data fields and episode behavior
    (`@vf.reward`/`@vf.metric`/`@vf.stop` methods, lifecycle hooks, `load_tools`/`load_user`).
    Data gets the plain name, constructors get the `load_` prefix — so a `tools`/`user`
    config field and its loader coexist (same convention as `Taskset.load_tasks`)."""

    model_config = ConfigDict(frozen=True)

    NEEDS_CONTAINER: ClassVar[bool] = False
    """Whether this task only runs in a container runtime (docker/prime). When True the
    subprocess runtime is refused — for tasks whose work only makes sense inside a
    per-task image (e.g. a SWE repo sandbox)."""
    STATE: ClassVar[type[State]] = State
    """The per-rollout `State` type this task's episodes carry (`trace.state`, shared with
    tool/user servers and read+written by scoring). Override with a `State` subclass to
    type it; the base (empty) `State` by default."""

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
    (`user`) then opens the conversation, its first `respond` supplying the initial user
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
    sources: tuple[str, ...] = ()
    """Trace ids this task was derived from. Plain agent programs may stamp lineage here;
    `Agent.run(parents=...)` remains the canonical explicit graph edge."""
    relation: str | None = None
    """Free-form relationship to `sources` (for example "judges" or "solves")."""

    @property
    def prompt_text(self) -> str:
        """The prompt as plain text — `prompt` itself when it's a `str`, the joined text
        content of a `Messages` prompt (images are dropped), `""` when the task carries no
        prompt. For consumers that need the task's framing as text regardless of the prompt
        form, e.g. a judge's prompt template."""
        if isinstance(self.prompt, str):
            return self.prompt
        texts = [content_text(message.content) for message in self.prompt or []]
        return "\n\n".join(text for text in texts if text)

    # --- episode behavior — override on subclasses; the instance is still pure data ------

    def load_tools(self) -> "list[Toolset]":
        """Tool servers exposing this task's tools to the model — `vf.Toolset`s (classes with
        `@vf.tool` methods), each carrying its `config` (placement / runtime; a remote `url`
        for an already-running server). Empty by default; override to give a task tools."""
        return []

    def load_user(self) -> "User | None":
        """A user simulator for this task — structurally a tool server (an MCP server
        with a runtime), but driven by the framework, not exposed to the model. After
        each model turn the interception server calls its `respond` tool and injects the
        reply as a user turn. None by default; override to make a task a simulated
        multi-turn conversation (e.g. a TextArena game)."""
        return None

    async def setup(self, trace: "Trace", runtime: "Runtime") -> None:
        """Prepare the live runtime for this task, after `runtime.start()` and before the
        harness runs. No-op by default; override to run per-task setup in the runtime (e.g.
        a SWE row checking out its base commit). Errors propagate and fail the rollout.

        Like the scoring methods, hooks declare the inputs they need *by parameter name* and
        the framework injects them: any subset of `task` (self, for signature symmetry),
        `trace`, `runtime`. The trace (and its per-rollout `trace.state`) already exists when
        `setup` runs, so an override may stash per-rollout state there."""
        return None

    async def finalize(self, trace: "Trace", runtime: "Runtime") -> None:
        """Post-process the live runtime after the harness finishes, before scoring. No-op
        by default; override to do per-rollout work the rewards (or a downstream agent's
        task) depend on — apply/commit the agent's diff, run a build, scrape runtime
        artifacts into `trace.info`. Runs while the runtime is still live (after generation,
        before `@reward`/`@metric`); the symmetric counterpart to `setup`. Errors propagate
        and fail the rollout."""
        return None

    async def validate(self, runtime: "Runtime") -> bool:
        """Check this task is well-formed and solvable, independent of any model rollout —
        run by the `validate` entrypoint, never during a rollout. Valid (True) by default;
        override to assert the ground truth holds (e.g. a SWE row applying its gold patch and
        running its tests, or gsm8k confirming the verifier accepts the gold answer). Runs in
        a live runtime started for the task with `setup` already applied (a pure-data check
        can ignore it). Return False — or raise — to mark the task invalid; the entrypoint
        records the reason (the raised error's message)."""
        return True

    async def score(self, trace: "Trace", runtime: "Runtime | None" = None) -> None:
        """Score one rollout: run all `@metric`, then `@reward` methods over its trace,
        concurrently within each phase. Each metric is recorded in `trace.metrics` (a
        number, or a mapping merged in); each reward (weighted — likewise a number or a
        mapping merged in) in `trace.rewards`, which `trace.reward` sums. Methods declare
        what they need — `task` (self), `trace`, `runtime` — so a reward is either a pure
        function of the trace or runs read/write/exec in that (still-live) runtime, e.g. a
        verifier script. `runtime` may be None for offline replay; methods that require a
        runtime are skipped, while trace-only methods still re-score."""
        available = {"task": self, "trace": trace}
        if runtime is not None:
            available["runtime"] = runtime

        def can_run(fn) -> bool:
            if runtime is not None:
                return True
            param = inspect.signature(fn).parameters.get("runtime")
            return param is None or param.default is not inspect.Parameter.empty

        async with boundary(TaskError, f"task {type(self).__name__} scoring"):
            for kind in ("metric", "reward"):
                fns = [fn for fn in discover_decorated(self, kind) if can_run(fn)]
                results = (
                    [await invoke(fn, available) for fn in fns]
                    if len(fns) < 2
                    else await asyncio.gather(*(invoke(fn, available) for fn in fns))
                )
                for fn, result in zip(fns, results):
                    if kind == "metric":
                        if isinstance(result, Mapping):
                            trace.record_metrics(result)
                        else:
                            trace.record_metric(fn.__name__, result)
                    else:
                        weight = getattr(fn, "_vf_weight", 1.0)
                        if isinstance(result, Mapping):
                            for name, value in result.items():
                                trace.record_reward(name, value, weight)
                        else:
                            trace.record_reward(fn.__name__, result, weight)

    async def score_group(self, traces: "list[Trace]") -> None:
        """Score a group of rollouts of this task: run every `@group_reward` over all the
        traces at once (pairwise/preference rewards), each returning one score per trace,
        aligned to `traces`. A group reward declares what it needs — `task` (self) and
        `traces` — and compares trace metadata (anything from the runtime is recorded per
        rollout as a `@metric` first). Scores are weighted into each trace's reward,
        alongside the per-rollout rewards. No-op without `@group_reward`s."""
        rewards = discover_decorated(self, "group_reward")
        if not rewards:
            return
        available = {"task": self, "traces": traces}
        async with boundary(TaskError, f"task {type(self).__name__} group scoring"):
            results = (
                [await invoke(fn, available) for fn in rewards]
                if len(rewards) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in rewards))
            )
            for fn, scores in zip(rewards, results):
                weight = getattr(fn, "_vf_weight", 1.0)
                for trace, score in zip(traces, scores):
                    trace.record_reward(fn.__name__, score, weight)


class WireTask(Task):
    """A `Task` that accepts (and preserves) task-specific extra fields. Lets a `Trace`
    be typed on the wire — `Trace[WireTask]` — without importing the task's package, since the
    real `Task` subclass's extra fields land in `model_extra` instead of being rejected. A
    caller that imports the package upgrades to the real type via `task_type(taskset_id)`."""

    model_config = ConfigDict(extra="allow")


TaskT = TypeVar("TaskT", bound=Task)
