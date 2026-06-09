"""The environment: a taskset composed with an harness and a runtime.

The Environment is the eval-level composition and *resolver* — it does not itself run
rollouts. It holds the taskset, harness, runtime config, and timeouts; lists the tasks;
and turns one task into a runnable `Episode` of `n` `Rollout`s, resolving per task the
runtime (image + resources, with cli/task/default precedence) and the timeouts. Execution
lives one level down: an `Episode` runs `n` `Rollout`s of a task and scores them
(per-rollout `@reward`/`@metric`, then cross-rollout `@group_reward`); each `Rollout`
runs one trajectory. The taskset's `@reward`/`@metric` get the rollout's runtime
(read/exec inside it), so a task scores correctly under any harness; `@group_reward`s
compare a task's rollouts.
"""

import contextlib
import logging

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.harness import HarnessConfig
from verifiers.v1.clients import RolloutContext
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.episode import Episode
from verifiers.v1.interception import RolloutLimits
from verifiers.v1.retries import RetryConfig
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig
from verifiers.v1.tools import serve_shared


class TimeoutConfig(BaseConfig):
    """Framework-enforced wall-clock timeouts, in seconds (None = no limit)."""

    rollout: float | None = None
    """Max wall-clock for the rollout (the harness run)."""
    scoring: float | None = None
    """Max wall-clock for scoring — verify + rewards/metrics."""


class EnvConfig(BaseConfig):
    """The rollout's two peers: the taskset (data + scoring) and the harness (which
    program drives it, and where it runs — `harness.runtime`). Both are chosen at eval
    time, not by the env — only `taskset` is narrowed per env (to its config type,
    inferred from `load_taskset`). Tool-server placement lives on `taskset.tools`."""

    # SerializeAsAny: these hold resolved subclasses (e.g. MathConfig, DefaultHarnessConfig);
    # without it model_dump() narrows to the base type and drops the subclass fields, so the
    # env-server subconfig the orchestrator writes would lose taskset/harness-specific knobs.
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    timeout: TimeoutConfig = TimeoutConfig()
    retry: RetryConfig = RetryConfig()
    max_turns: int | None = None
    """Max model turns per rollout (None = no limit). Enforced by the framework (the
    interception server refuses turns past it), so it applies to any harness — turn
    capping is a framework concern, never an harness or task field."""
    max_input_tokens: int | None = None
    """Max input (prompt) tokens per rollout (None = no limit). Caps the trace's
    `prompt_len`; framework-enforced between turns."""
    max_output_tokens: int | None = None
    """Max output (completion) tokens per rollout (None = no limit). Caps the trace's
    `completion_len`; framework-enforced between turns."""
    max_total_tokens: int | None = None
    """Max total (prompt + completion) tokens per rollout (None = no limit). Caps the
    trace's `total_tokens`; framework-enforced between turns."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_plugins(cls, data):
        """Resolve the generic `taskset` / `harness` to its specific config type by `id`, so
        env-specific fields validate against the real plugin config (no untyped args dict)."""
        from verifiers.v1.loaders import harness_config_type, taskset_config_type

        for field, resolve, default_id in (
            ("taskset", taskset_config_type, None),
            ("harness", harness_config_type, "default"),
        ):
            raw = data.get(field)
            if isinstance(raw, BaseConfig):
                raw = raw.model_dump()
            raw = dict(raw or {})
            ident = raw.get("id") or default_id
            if ident:
                data[field] = resolve(ident).model_validate({**raw, "id": ident})
        return data


logger = logging.getLogger(__name__)


class Environment:
    def __init__(self, config: EnvConfig) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        self.config = config
        self.taskset = load_taskset(config.taskset)
        self.harness = load_harness(config.harness)
        self.harness_timeout = config.timeout.rollout
        self.scoring_timeout = config.timeout.scoring
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self._warned_resources: set[tuple[str, str]] = set()

    def runtime_for(self, task: Task) -> RuntimeConfig:
        """Resolve the runtime config for a task: inject the task's `image` (a task
        with an image must run in a container — refuse subprocess), and apply the
        task's requested `resources` to fields the runtime supports. Precedence is
        cli/toml > task > default; a resource the runtime doesn't support warns once."""
        config = self.harness.config.runtime
        updates: dict = {}
        if task.image is not None:
            if isinstance(config, SubprocessConfig):
                raise ValueError(
                    f"task {task.idx!r} requires image {task.image!r}, but the subprocess "
                    "runtime has no container; use the docker or prime runtime"
                )
            updates["image"] = task.image
        for field, value in task.resources.model_dump(exclude_none=True).items():
            spec = type(config).model_fields.get(field)
            if spec is None:
                key = (config.type, field)
                if key not in self._warned_resources:
                    self._warned_resources.add(key)
                    logger.warning(
                        "runtime %r doesn't support resource %r; ignoring it",
                        config.type,
                        field,
                    )
            elif (
                getattr(config, field) == spec.default
            ):  # still the default → task may set it
                updates[field] = value
            # else: cli/toml changed it from the default → it wins over the task
        return config.model_copy(update=updates) if updates else config

    def episode(self, task: Task, ctx: RolloutContext, n: int = 1) -> Episode:
        """Resolve `task` into a runnable episode of `n` rollouts: pick its runtime
        (image + resources) and its timeouts (cli/toml > task > default, None = no limit),
        build one `Rollout` per sample sharing them, and wrap them in an `Episode` (which
        runs them and applies the taskset's `@group_reward`s across their traces).

        A taskset with `@group_reward`s compares a task's rollouts, so it needs >=2 of
        them — refuse `n < 2` there (rather than silently scoring a group of one)."""
        if n < 2 and discover_decorated(self.taskset, "group_reward"):
            raise ValueError(
                f"taskset defines @group_reward(s), which compare a task's rollouts and "
                f"need >=2; got n={n} (pass -r/--num-rollouts >= 2)"
            )
        runtime_config = self.runtime_for(task)
        harness_timeout = (
            self.harness_timeout
            if self.harness_timeout is not None
            else task.harness_timeout
        )
        scoring_timeout = (
            self.scoring_timeout
            if self.scoring_timeout is not None
            else task.scoring_timeout
        )
        rollouts = [
            Rollout(
                task=task,
                taskset=self.taskset,
                harness=self.harness,
                ctx=ctx,
                runtime_config=runtime_config,
                harness_timeout=harness_timeout,
                scoring_timeout=scoring_timeout,
                limits=self.limits,
            )
            for _ in range(n)
        ]
        return Episode(rollouts, self.taskset, retry=self.config.retry)

    @contextlib.asynccontextmanager
    async def shared_tools(self, tasks: list[Task]):
        """When `tools.shared` is set, start the taskset's tool servers ONCE for the eval
        (in their own `tools.runtime`) and yield `{name: url}` to inject into every
        rollout — so an expensive corpus is built once, not per rollout. No-op ({}) when
        not shared. Shared servers must be task-agnostic, so they're read off any task."""
        tools = self.taskset.config.tools
        if not (tools.shared and tasks):
            yield {}
            return
        async with serve_shared(
            self.taskset.tool_servers(tasks[0]), tools.runtime
        ) as urls:
            yield urls
