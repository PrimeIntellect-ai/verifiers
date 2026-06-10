"""The taskset: produces typed tasks and owns scoring.

A `Taskset` is the data + judgement half of an environment. It yields typed
`Task`s, may expose tools via `tools`, and defines rewards/metrics as
decorated methods. All task framing lives in each task's user prompt (baked in by
`load_tasks`); the harness drives control flow.

It is the single judgement authority, scored at two granularities (execution lives in
the Rollout — per-rollout — and the Episode — group — which call these):
  - `score` runs `@reward`/`@metric` over one trace (in its live runtime).
  - `score_group` runs `@group_reward` over all the rollouts of one task at once —
    pairwise/preference rewards that compare samples.

For a heterogeneous taskset (different verification per task), have a single
`@reward` branch on a typed task field.
"""

import asyncio
from collections.abc import Mapping
from typing import Generic, TypeVar

from pydantic import model_validator
from pydantic_config import BaseConfig

from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.ids import EnvId, env_name
from verifiers.v1.runtimes import Runtime, RuntimeConfig, SubprocessConfig
from verifiers.v1.task import TaskT
from verifiers.v1.tools import Tools
from verifiers.v1.trace import Trace
from verifiers.v1.user import User


class ToolsConfig(BaseConfig):
    """How a taskset's `tools` are run (`colocated` and `shared` are mutually
    exclusive; reachability — localhost vs tunnel — is then inferred):
      - colocated: in the harness's runtime, per rollout (localhost; `runtime` ignored).
      - shared:    one instance for the whole eval, in its own `runtime`.
      - neither:   its own `runtime`, per rollout."""

    colocated: bool = True
    """Run each tool server inside the harness's runtime (localhost, per rollout). The
    default: a self-contained uv-script server runs anywhere. A data-heavy server that
    re-fetches per rollout should opt out (host-served, or `shared`)."""
    shared: bool = False
    """Run one tool-server instance for the whole eval, shared across rollouts (in its
    own `runtime`). Mutually exclusive with `colocated`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """The tool server's own runtime, used when not colocated (colocated uses the
    harness's runtime)."""

    @model_validator(mode="after")
    def _exclusive(self) -> "ToolsConfig":
        if self.colocated and self.shared:
            raise ValueError("tools.colocated and tools.shared are mutually exclusive")
        return self


class TasksetConfig(BaseConfig):
    """Base taskset config. Subclass to add task-generation knobs."""

    id: EnvId = ""
    """The taskset id, which selects this taskset: a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see
    `EnvId`). Set via `--taskset.id`."""
    tools: ToolsConfig = ToolsConfig()

    @property
    def name(self) -> str:
        """The taskset's package name (the id with any org / version stripped)."""
        return env_name(self.id)


ConfigT = TypeVar("ConfigT", bound=TasksetConfig)


class Taskset(Generic[TaskT, ConfigT]):
    """Generic over its task and config types, so `self.config` and `load_tasks`
    are fully typed. Subclass: implement `load_tasks`, add @reward/@metric."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def load_tasks(self) -> list[TaskT]:
        raise NotImplementedError

    def tools(self, task: TaskT) -> list[Tools]:
        """MCP servers exposing this task's tools, launched in the runtime by the
        harness. Empty by default; override to give a task tools."""
        return []

    def user(self, task: TaskT) -> User | None:
        """A user simulator for this task — structurally a tool server (an MCP server
        with a runtime), but driven by the framework, not exposed to the model. After
        each model turn the interception server calls its `respond` tool and injects the
        reply as a user turn. None by default; override to make a task a simulated
        multi-turn conversation (e.g. a TextArena game)."""
        return None

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        """Score one rollout: run all `@metric` then `@reward` over its trace,
        concurrently within each phase. Each metric is recorded in `trace.metrics`
        (a number, or a mapping merged in); each reward (weighted) in `trace.rewards`,
        which `trace.reward` sums. Signals declare what they need — `task`, `trace`,
        `runtime` — so a reward is either a pure function of the trace or runs
        read/write/exec in that (still-live) runtime, e.g. a verifier script."""
        available = {"task": trace.task, "trace": trace, "runtime": runtime}
        metrics = discover_decorated(self, "metric")
        for fn, result in zip(
            metrics, await asyncio.gather(*(invoke(fn, available) for fn in metrics))
        ):
            if isinstance(result, Mapping):
                trace.record_metrics(result)
            else:
                trace.record_metric(fn.__name__, result)
        rewards = discover_decorated(self, "reward")
        for fn, result in zip(
            rewards, await asyncio.gather(*(invoke(fn, available) for fn in rewards))
        ):
            trace.record_reward(fn.__name__, result, getattr(fn, "_vf_weight", 1.0))

    async def score_group(self, traces: list[Trace]) -> None:
        """Score a group of rollouts of one task: run every `@group_reward` over all
        the traces at once (pairwise/preference rewards), each returning one score per
        trace, aligned to `traces`. A group reward declares what it needs — `task` (the
        shared task) and `traces` — and compares trace metadata (anything from the
        runtime is recorded per rollout as a `@metric` first). Scores are weighted into
        each trace's reward, alongside the per-rollout rewards. No-op without `@group_reward`s."""
        rewards = discover_decorated(self, "group_reward")
        if not rewards:
            return
        available = {"task": traces[0].task, "traces": traces}
        for fn, scores in zip(
            rewards, await asyncio.gather(*(invoke(fn, available) for fn in rewards))
        ):
            weight = getattr(fn, "_vf_weight", 1.0)
            for trace, score in zip(traces, scores):
                trace.record_reward(fn.__name__, score, weight)
