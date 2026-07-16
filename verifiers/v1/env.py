"""Compose a taskset, its agents, and how one task becomes one env-rollout."""

import asyncio
import contextlib
import logging
import traceback
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Generic, Literal, TypeVar

from pydantic import Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.clients import Client, ClientConfig, ModelContext, resolve_client
from verifiers.v1.types import ID, SamplingConfig
from verifiers.v1.interception import (
    ElasticInterceptionPoolConfig,
    Interception,
    InterceptionConfig,
    make_interception,
    requires_tunnel,
)
from verifiers.v1.session import Respond, RolloutLimits
from verifiers.v1.retries import RetryConfig, run_record_with_retry
from verifiers.v1.runtimes import (
    RuntimeConfig,
    SubprocessConfig,
    runtime_is_local,
)
from verifiers.v1.task import Task, resolve_server_config
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Error, RolloutRecord, Trace, TraceTask
from verifiers.v1.utils.generic import generic_type
from verifiers.v1.utils.memory import trim_memory_periodically
from verifiers.v1.mcp import SharedToolServer, serve_shared

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent
    from verifiers.v1.runtimes import Runtime


class TimeoutConfig(BaseConfig):
    """Framework-enforced wall-clock timeouts per rollout stage, in seconds (None = no
    limit). Each bounds one stage of `run_rollout`: task and harness setup, the harness
    run, the task's `finalize` hook, then scoring. `score` is the one env-level stage:
    the cross-trace `Environment.score` hook, run once per env-rollout after its traces
    finish."""

    setup: float | None = None
    """Shared wall-clock budget for task setup and harness provisioning."""
    rollout: float | None = None
    """Max wall-clock for the rollout (the harness run)."""
    finalize: float | None = None
    """Max wall-clock for the task's `finalize` hook (post-run work, before scoring)."""
    scoring: float | None = None
    """Max wall-clock for task and harness scoring."""
    score: float | None = None
    """Max wall-clock for the env's `score()` hook (cross-trace judgement over a
    finished env-rollout — per-trace task/harness scoring is bounded by `scoring`)."""


class AgentConfig(BaseConfig):
    """One env role: the agent that fills a seat in `Environment.roles()`. Everything
    defaults to the run's own settings, so `AgentConfig()` is "the policy under
    evaluation/training" — a role pins only what makes it a different seat (its own
    harness, a frozen model, an off-train endpoint, tighter limits)."""

    harness: SerializeAsAny[HarnessConfig] | None = None
    """The role's program + runtime policy (None = the run's `--harness.*` — late
    binding, like `model`/`client`/`sampling`, so pairing an env never silently
    swaps the policy's harness). Pin it to seat the role on its own program (e.g.
    `vf.HarnessConfig(id="direct")` for a bare in-process chat seat)."""
    model: str | None = None
    """Model id (None = the run's model — late binding: the role is played by the
    policy being evaluated or trained, which is what makes self-play trainable)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client). Set it to route a fixed role (a
    frozen judge, a pinned user sim) off the training endpoint."""
    sampling: SamplingConfig | None = None
    """Sampling override (None = the run's sampling)."""
    max_turns: int | None = None
    """Max model turns per rollout for this role (None = the env's)."""
    max_input_tokens: int | None = None
    """Max input tokens per rollout for this role (None = the env's)."""
    max_output_tokens: int | None = None
    """Max output tokens per rollout for this role (None = the env's)."""
    max_total_tokens: int | None = None
    """Max total tokens per rollout for this role (None = the env's)."""
    trainable: bool = True
    """Stamped on every trace this role produces: whether its tokens are training data
    for the run's policy (set False for fixed-model roles)."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow a *pinned* `harness` to its specific config type by `id` (the same
        resolution as `EnvConfig`), so role-harness fields validate typed. An absent
        harness stays None — the run's harness, resolved at env construction."""
        from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

        if isinstance(data, dict) and data.get("harness") is not None:
            narrow_plugin_field(data, "harness", harness_config_type, "default")
        return data


def _deep_merge(base: dict, override: dict) -> dict:
    """`override` onto `base`, recursing into dicts — so a partial nested override
    (e.g. a role's `{"sampling": {"temperature": 0.2}}`) keeps the untouched keys of
    the declared default. An override that switches a subtree's discriminator (`id`/
    `type`: a different harness, runtime, or judge spec) replaces the subtree
    wholesale — the old plugin's fields must not leak into the new type's validation."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            switched = any(
                k in value and k in merged[key] and value[k] != merged[key][k]
                for k in ("id", "type")
            )
            merged[key] = value if switched else _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class EnvParams(BaseConfig):
    """An environment's own parameters: the typed block a multi-agent env declares —
    its roles as `AgentConfig` fields, plus any env-level knobs — living under
    `--env.*` / `[env]` on every run config (`--env.user.model`, `--env.attempts 8`).
    Subclass it next to your `Environment` subclass and bind it via
    `Environment[YourParams]`; the framework narrows the `EnvConfig.env` field to it
    by the env `id` (else the taskset id), exactly like `taskset`/`harness`. The base
    is empty — the single-agent case has no params."""

    id: ID = ""
    """Which `Environment` runs the taskset — the control flow between agents. Empty
    (the default) keeps the taskset's own story: the `Environment` subclass its package
    exports, else the single-agent base. Set it to pair a reusable interaction with any
    taskset (`--env.id best-of-n --taskset.id gsm8k-v1`): a bundled env
    (`verifiers.v1.envs`), a local package, or a Hub `org/name[@version]`. An explicit
    id wins over the taskset's bundled env."""

    @model_validator(mode="before")
    @classmethod
    def _merge_role_defaults(cls, data):
        """A role's declared default (`user: AgentConfig = AgentConfig(model=...,
        trainable=False)`) is a field-default *instance*, which plain validation would
        replace wholesale on any partial override — `--env.user.sampling.temperature`
        must not silently reset the pinned model and trainability. Deep-merge the
        partial data over the declared default, so an override touches exactly the
        keys it names."""
        if isinstance(data, dict):
            for name, field in cls.model_fields.items():
                if isinstance(field.default, AgentConfig) and isinstance(
                    data.get(name), dict
                ):
                    data[name] = _deep_merge(
                        field.default.model_dump(exclude_none=True), data[name]
                    )
        return data


class StaticPoolConfig(BaseConfig):
    """Fixed env-server pool: pre-spawn `num_workers` workers up front."""

    type: Literal["static"] = "static"
    num_workers: int = Field(4, ge=1)
    """Worker processes to pre-spawn (1 = a single in-process server, no pool)."""


class ElasticPoolConfig(BaseConfig):
    """Elastic env-server pool: start at one worker and scale up on demand."""

    type: Literal["elastic"] = "elastic"
    max_workers: int | None = None
    """Upper bound on workers (None = unbounded)."""
    multiplex: int = Field(128, ge=1)
    """Rollouts per worker for the scale-up trigger: add a worker once in-flight rollouts
    reach 90% of `workers * multiplex`."""


# Discriminated on `type` so the CLI selects with `--pool.type static|elastic`.
PoolConfig = Annotated[
    StaticPoolConfig | ElasticPoolConfig, Field(discriminator="type")
]


def pool_serve_kwargs(pool: StaticPoolConfig | ElasticPoolConfig) -> dict:
    """Unpack a pool config into `serve_env` kwargs (`max_workers` / `multiplex` / `elastic`)."""
    if isinstance(pool, ElasticPoolConfig):
        return {
            "max_workers": pool.max_workers,
            "multiplex": pool.multiplex,
            "elastic": True,
        }
    return {"max_workers": pool.num_workers, "elastic": False}


class EnvConfig(BaseConfig):
    """The taskset that loads tasks and the harness that runs them."""

    # SerializeAsAny: these hold resolved subclasses (e.g. MathConfig, DefaultHarnessConfig,
    # a multi-agent env's params); without it model_dump() narrows to the base type and drops
    # the subclass fields, so the env-server subconfig the orchestrator writes would lose
    # taskset/harness/env-specific knobs.
    taskset: SerializeAsAny[TasksetConfig] = TasksetConfig()
    harness: SerializeAsAny[HarnessConfig] = HarnessConfig(id="default")
    env: SerializeAsAny[EnvParams] = EnvParams()
    """The environment — the control flow between agents — and its parameters
    (`EnvParams`): which env runs the taskset (`--env.id`, empty = the taskset's own
    story) plus its roles and knobs, narrowed to the env's declared params type by the
    env id, else the taskset id (`--env.user.model`, `[env]` in TOML). Empty for plain
    tasksets."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    max_turns: int | None = None
    """Max model turns per rollout (None = no limit). Enforced by the framework (the
    interception server refuses turns past it), so it applies to any harness — turn
    capping is a framework concern, never an harness or task field."""
    max_input_tokens: int | None = None
    """Max input (prompt) tokens per rollout (None = no limit). Caps the trace's
    `num_input_tokens`; framework-enforced between turns."""
    max_output_tokens: int | None = None
    """Max output (completion) tokens per rollout (None = no limit). Caps the trace's
    `num_output_tokens`; framework-enforced between turns."""
    max_total_tokens: int | None = None
    """Max total (prompt + completion) tokens per rollout (None = no limit). Caps the
    trace's `num_total_tokens`; framework-enforced between turns."""
    interception: InterceptionConfig = ElasticInterceptionPoolConfig()
    """The interception shape (see `verifiers.v1.interception`): `elastic` (the
    default — servers grown on demand, `multiplex` rollouts each), `server` (one server,
    with a tunnel choice incl. a bring-your-own endpoint), or `static` (a fixed list of
    such servers)."""
    # --- legacy (v0) backwards-compat -----------------------------------------
    id: ID | None = None
    """Classic (v0) env id (`name`, `org/name`, or `org/name@version` — installed from the
    hub on demand), loaded via `verifiers.load_environment` and run through the legacy
    bridge. Set this *instead of* `taskset` to run a v0 environment."""
    args: dict = {}
    """Construction kwargs forwarded to `load_environment(id, **args)`."""
    extra_env_kwargs: dict = {}
    """Post-load kwargs applied to the v0 env via `env.set_kwargs(**extra_env_kwargs)` (e.g.
    `max_total_completion_tokens`, `max_seq_len`, `timeout_seconds`) — typically
    auto-populated by the orchestrator, distinct from the `args` passed at construction."""

    @property
    def is_legacy(self) -> bool:
        """A v0/legacy env (run via the bridge): a legacy `id` is set and no v1 `taskset`."""
        return self.id is not None and not self.taskset.id

    @property
    def env_id(self) -> str:
        """The run's identifier — the v1 taskset id (prefixed by the selected env when
        `--env.id` pairs one in, so `best-of-n+gsm8k-v1` and a plain `gsm8k-v1` run
        stay distinguishable on records and output dirs), else the legacy v0 env id."""
        if self.taskset.id and self.env.id:
            return f"{self.env.id}+{self.taskset.id}"
        return self.taskset.id or self.id or ""

    # --- end legacy -----------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _resolve_plugins(cls, data):
        """Resolve the generic `taskset` / `harness` / `env` to its specific config type,
        so env-specific fields validate against the real plugin config (no untyped args
        dict) — taskset/harness by their `id`, the env params by the taskset id (a
        taskset shipping an `Environment` subclass declares its params type via
        `Environment[YourParams]`)."""
        from verifiers.v1.loaders import (
            default_harness_id,
            env_params_type,
            harness_config_type,
            narrow_plugin_field,
            taskset_config_type,
        )

        narrow_plugin_field(data, "taskset", taskset_config_type)
        taskset = data.get("taskset")
        taskset_id = (
            taskset.get("id")
            if isinstance(taskset, dict)
            else getattr(taskset, "id", None)
        )
        # A taskset that bundles its own harness runs with it by default; an explicit
        # `--harness.id` / toml id (already on the field) takes precedence.
        narrow_plugin_field(
            data, "harness", harness_config_type, default_harness_id(taskset_id or "")
        )
        raw = data.get("env")
        env_id = (
            getattr(raw, "id", "")
            if isinstance(raw, BaseConfig)
            else (raw or {}).get("id", "")
            if isinstance(raw, dict)
            else ""
        )
        params_cls = env_params_type(taskset_id or "", env_id or "")
        if isinstance(raw, EnvParams) and isinstance(raw, params_cls):
            pass  # already at least as specifically typed (e.g. programmatic) — keep
        else:
            if isinstance(raw, BaseConfig):
                raw = raw.model_dump()
            data["env"] = params_cls.model_validate(raw or {})
        return data


class EnvServerConfig(EnvConfig):
    """An env plus how it's *served*: the env definition (`EnvConfig`) and the worker-pool
    sizing. Shared by the `serve` CLI, server-backed eval, and prime-rl's orchestrator, so
    they all configure the pool the same way (`--pool.type elastic|static`)."""

    pool: PoolConfig = ElasticPoolConfig()
    """Worker-pool sizing for the env server. `elastic` (default) starts at one worker and
    scales up on demand; `static` pre-spawns a fixed `num_workers`."""


logger = logging.getLogger(__name__)


def resolve_runtime_config(
    base: RuntimeConfig, task: Task, warned: set[tuple[str, str]] | None = None
) -> RuntimeConfig:
    """Resolve a task's runtime config from a `base`: inject the task's `image` (a task with
    an image must run in a container — refuse subprocess), and apply its `workdir` and
    requested `resources` to the fields the runtime supports. Precedence is cli/toml > task >
    default; a resource the runtime doesn't support warns once (deduped via `warned`). Shared
    by `Agent.run` (every rollout resolves through it) and the `validate` entrypoint."""
    config = base
    updates: dict = {}
    if task.data.image is not None:
        if isinstance(config, SubprocessConfig):
            raise ValueError(
                f"task {task.data.idx!r} requires image {task.data.image!r}, but the subprocess "
                "runtime has no container; use the docker or prime runtime"
            )
        updates["image"] = task.data.image
    workdir_spec = type(config).model_fields.get("workdir")
    if (
        task.data.workdir is not None
        and workdir_spec is not None
        and getattr(config, "workdir") == workdir_spec.default
    ):
        updates["workdir"] = task.data.workdir
    for resource, value in task.data.resources.model_dump(exclude_none=True).items():
        spec = type(config).model_fields.get(resource)
        if spec is None:
            key = (config.type, resource)
            if warned is not None and key not in warned:
                warned.add(key)
                logger.warning(
                    "runtime %r doesn't support resource %r; ignoring it",
                    config.type,
                    resource,
                )
        elif (
            getattr(config, resource) == spec.default
        ):  # still the default → task may set it
            updates[resource] = value
        # else: cli/toml changed it from the default → it wins over the task
    return config.model_copy(update=updates) if updates else config


def validate_pairing(
    harness: Harness,
    task_cls: type[Task],
    runtime_config: RuntimeConfig,
    *,
    shared_tools: Collection = (),
) -> None:
    """Reject an impossible harness/task/runtime combination before any work happens.
    Every check reads class-level facts (`Task.tools` / `NEEDS_CONTAINER`, plus
    whatever shared MCP the caller brings), so a failure here holds for
    every row the task class can carry. Shared by `Environment` (once, at init, with the
    task type read off the `Taskset[TaskT, ...]` generic and the taskset's declared
    `tools` — on the env server it fails worker startup instead of every request) and
    `Agent.run` (per run, with the concrete task's type and the agent's borrowed
    `shared_tools` servers, against the resolved runtime). Only the collection's
    emptiness matters — declarations and live servers alike mean MCP is in play.
    (Hosting a user is run-scoped, not task-scoped — `Agent.run` checks
    `SUPPORTS_USER_SIM` when a `user=` is actually passed.)"""
    if not harness.SUPPORTS_MCP and (task_cls.tools or shared_tools):
        raise ValueError(
            f"Harness {harness.config.id!r} does not support MCP tools, but "
            f"{task_cls.__name__} exposes tool servers (MCP). Run it with a harness that "
            f"supports MCP (e.g. --harness.id default), or use tasks without tools."
        )
    if task_cls.NEEDS_CONTAINER and isinstance(runtime_config, SubprocessConfig):
        raise ValueError(
            f"{task_cls.__name__} needs a container runtime (NEEDS_CONTAINER), but "
            "this run resolves to the subprocess runtime; use --harness.runtime.type "
            "docker or prime."
        )


def cap_remote_harness_timeout(
    harness_timeout: float | None, runtime_config: RuntimeConfig, task: Task
) -> float | None:
    """Remote sandboxes have a maximum lifetime of 24 hours: cap the harness timeout
    there (with a warning) so a long run times out cleanly in the framework instead of
    the provider killing the box mid-run. Shared by the env's rollouts and
    `Agent.run`."""
    if (
        harness_timeout is not None
        and harness_timeout > 24 * 60 * 60
        and not runtime_is_local(runtime_config)
    ):
        logger.warning(
            "task %r resolves to a %.1f-hour harness timeout, but %s sandboxes have a "
            "maximum lifetime of 24 hours; capping it at 24 hours",
            task.data.idx,
            harness_timeout / (60 * 60),
            runtime_config.type,
        )
        return 24 * 60 * 60
    return harness_timeout


@dataclass
class RunSlot:
    """One planned env-rollout of a task, observable while it happens: `traces`
    collects the current attempt's live traces from the moment the engine mints them
    (a retry restarts the list with the fresh attempt's; a single-agent rollout has
    exactly one), `record` is the finished rollout's record, and `done` flips once
    that record is final. The `--rich` dashboard renders slots (deriving each trace's
    live stage from its timing spans); `--resume` preloads the previous session's kept
    records as `finished` slots."""

    task: Task
    traces: list[Trace] = field(default_factory=list)
    record: RolloutRecord | None = None
    done: bool = False

    @classmethod
    def finished(cls, record: RolloutRecord) -> "RunSlot":
        return cls(
            task=Task(record.task.data),
            traces=list(record.traces),
            record=record,
            done=True,
        )


ParamsT = TypeVar("ParamsT", bound=EnvParams)


class Environment(Generic[ParamsT]):
    """A taskset, the agents that play it, and how one task becomes one env-rollout.

    Concrete, and its defaults ARE the single-agent case: the base class runs every
    existing taskset exactly as `--taskset.*`/`--harness.*` configure it. A multi-agent
    env subclasses it (exported from the taskset's package, loaded like every plugin),
    declares its parameters as an `EnvParams` subclass bound via
    `Environment[YourParams]` (available as `self.params`, addressed as `--env.*`),
    and overrides up to three methods:

      - `roles()` — which agent fills which seat, one `AgentConfig` per role.
      - `rollout(task, agents)` — how the agents interact on one task: imperative
        Python over `Agent` values; the returned traces are the rollout's record.
      - `score(task, traces)` — sibling-dependent judgement over the finished set.

    plus `setup()`/`teardown()` for env-owned shared resources. The base owns the rest —
    taskset + serving resources, per-role agent construction, records, retries,
    persistence/resume, the serve protocol — so a subclass never touches machinery."""

    def __init__(self, config: EnvConfig) -> None:
        from verifiers.v1.loaders import load_harness, load_taskset

        self.config = config
        params_cls = (
            generic_type(type(self), EnvParams, origin=Environment) or EnvParams
        )
        if not isinstance(config.env, params_cls):
            raise TypeError(
                f"{type(self).__name__} declares Environment[{params_cls.__name__}], "
                f"but config.env is {type(config.env).__name__}; build the config "
                "naming this taskset (its validator narrows `env`), or pass "
                f"env={params_cls.__name__}(...) explicitly"
            )
        self.params: ParamsT = config.env  # type: ignore[assignment]
        self.taskset = load_taskset(config.taskset)
        self.harness = load_harness(config.harness)
        task_cls = generic_type(type(self.taskset), Task, origin=Taskset) or Task
        self._roles: dict[str, AgentConfig] = dict(self.roles())
        if not self._roles:
            raise ValueError(
                f"{type(self).__name__}.roles() returned no roles; an env needs at "
                "least one agent (the base default is a single 'main' role)"
            )
        # Traces are role-stamped except in the base single-agent shape, where there's
        # only one seat and `role=None` keeps the wire identical to a plain eval's.
        self._stamp_roles = list(self._roles) != ["main"]
        self._harnesses: dict[str, Harness] = {
            name: self.harness
            if spec.harness is None or spec.harness == config.harness
            else load_harness(spec.harness)
            for name, spec in self._roles.items()
        }
        warned: set[str] = set()
        for name, harness in self._harnesses.items():
            validate_pairing(
                harness,
                task_cls,
                harness.config.runtime,
                shared_tools=type(self.taskset).tools,
            )
            # The warning is about the *agent* running arbitrary code on the host: every harness
            # hands it local execution (bash/edit, or a CLI agent) except the tool-less `null`
            # chat loop, whose program only relays the model and remote MCP tools — so exempt
            # `null`, warn for the rest (once per distinct harness across roles).
            if (
                harness.config.id != "null"
                and isinstance(harness.config.runtime, SubprocessConfig)
                and harness.config.id not in warned
            ):
                warned.add(harness.config.id)
                logger.warning(
                    "Harness %r is running in the subprocess runtime on the local system. "
                    "Local files and settings may affect the evaluation; use subprocess only "
                    "for debugging. Use --harness.runtime.type docker or prime for an isolated "
                    "run.",
                    harness.config.id,
                )
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self._shared_tools: dict[str, SharedToolServer] = {}
        self._interception: Interception | None = None
        """Eval-level serving resources, live only inside `serving()`: shared tool servers
        ({name: SharedToolServer}) and the interception. The env's agents borrow them, so
        every run planned inside the context rides them — neither runner has to
        thread them through `run_slot`."""
        self._agents: dict[str, "Agent"] | None = None
        self._agents_ctx: ModelContext | None = None
        self._role_clients: dict[str, Client] = {}
        """Clients resolved for roles that pin their own endpoint (`AgentConfig.client`),
        cached by config and closed when `serving()` exits."""

    # --- the multi-agent surface (override these) ------------------------------

    def roles(self) -> Mapping[str, AgentConfig]:
        """Which agent fills which seat. Called once, at construction (read
        `self.params` — a typed `AgentConfig` field per role is the idiom, giving
        `--env.<role>.model` CLI/TOML addressing for free). Default: the single-agent
        case — one `"main"` role on the run's own harness (`--harness.*`)."""
        return {"main": AgentConfig()}

    async def rollout(self, task: Task, agents: Mapping[str, "Agent"]) -> list[Trace]:
        """One env-rollout: how the agents interact on `task`. Imperative Python over
        the handed-in agents — a loop is rounds, `asyncio.gather` is fan-out, a
        function from traces to task data is chaining; every trace comes from an
        `agents[...]` verb. The returned traces are the rollout's record, in this
        order. An agent-run failure is data on its trace (this hook decides what a
        failed participant means); an exception raised here is the env-rollout itself
        failing. Default: the single-agent case — `"main"` runs the task once."""
        return [await agents["main"].run(task)]

    async def score(self, task: Task, traces: list[Trace]) -> None:
        """Sibling-dependent judgement over one env-rollout's finished traces — attach
        via `trace.record_reward`/`record_metric`. Per-trace judgement already ran on
        each trace's own task (hooks + `@reward`/`@metric`, box-live, as in any eval);
        override this only for signals that need the finished sibling set (relative
        comparison, solve-rate over children, learnability). Bounded by
        `timeout.score`. Default: no-op."""

    def complete(self, record: RolloutRecord) -> bool:
        """Whether a finished env-rollout is a valid result — the verdict `--resume`
        reads to decide what to keep vs. redo. Default: `record.ok` (no rollout-level
        errors and no errored trace). An env whose `rollout()` deliberately tolerates
        a failed participant (a forfeited seat, a crashed editor) overrides this —
        typically `not record.errors` — else resume re-runs rollouts it already
        accepted. Read-only: what a failed participant *means* stays in `rollout()`
        and `score()`. (The server eval path keeps the strict default — its env lives
        in the workers.)"""
        return record.ok

    async def setup(self) -> None:
        """Bring up env-owned shared resources. Runs inside `serving()`, after the
        framework's resources (shared tool servers, interception) are live and before
        any rollout. Default: no-op."""

    async def teardown(self) -> None:
        """Tear down what `setup()` built. Runs when `serving()` exits, even when
        `setup()` failed partway — so it must tolerate partial state. Default: no-op."""

    # --- machinery (the base owns everything below) -----------------------------

    def _agents_for(self, ctx: ModelContext) -> dict[str, "Agent"]:
        """The agents that play this env's roles for `ctx`: per role, its harness and
        limits with the run's model/client/sampling unless the role pins its own,
        borrowing whatever serving resources (shared tool servers + interception) are
        live right now. Single-slot cache keyed by ctx value: the eval runner uses one
        ctx for the whole run (always hits), and the env server minting a ctx per
        request still hits (its clients are cached by config, so equal configs make
        equal ctxs). Constructing on a miss is cheap — agents are values."""
        from verifiers.v1.agent import (
            Agent,
        )  # env ↔ agent cycle, like `loaders` in init

        if self._agents is None or self._agents_ctx != ctx:
            self._agents = {
                name: Agent(
                    self._harnesses[name],
                    self._role_ctx(spec, ctx),
                    interception=self._interception,
                    shared_tools=self._shared_tools,
                    limits=self._role_limits(spec),
                    timeout=self.config.timeout,
                )
                for name, spec in self._roles.items()
            }
            self._agents_ctx = ctx
        return self._agents

    def _role_ctx(self, spec: AgentConfig, ctx: ModelContext) -> ModelContext:
        """The role's model leg: the run's `ctx` unless the role pins its own model,
        endpoint, or sampling (each falls back to the run's independently)."""
        if spec.model is None and spec.client is None and spec.sampling is None:
            return ctx
        return ModelContext(
            model=spec.model if spec.model is not None else ctx.model,
            client=self._client_for(spec.client)
            if spec.client is not None
            else ctx.client,
            sampling=spec.sampling if spec.sampling is not None else ctx.sampling,
        )

    def _client_for(self, config: ClientConfig) -> Client:
        """Resolve (and cache by config) a role-pinned endpoint's client; closed when
        `serving()` exits."""
        key = config.model_dump_json()
        if key not in self._role_clients:
            self._role_clients[key] = resolve_client(config)
        return self._role_clients[key]

    def _role_limits(self, spec: AgentConfig) -> RolloutLimits:
        """The role's per-rollout limits: each cap the role's own when set, else the
        env's."""
        return RolloutLimits(
            max_turns=spec.max_turns
            if spec.max_turns is not None
            else self.limits.max_turns,
            max_input_tokens=spec.max_input_tokens
            if spec.max_input_tokens is not None
            else self.limits.max_input_tokens,
            max_output_tokens=spec.max_output_tokens
            if spec.max_output_tokens is not None
            else self.limits.max_output_tokens,
            max_total_tokens=spec.max_total_tokens
            if spec.max_total_tokens is not None
            else self.limits.max_total_tokens,
        )

    async def run_record(
        self,
        task: Task,
        ctx: ModelContext,
        *,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> RolloutRecord:
        """One env-rollout of `task`, minted as the wire atom: run `rollout()` over the
        role agents, then `score()` over its traces (bounded by `timeout.score`).

        The handed-in agents are wrapped so every trace is stamped (`role`/`trainable`)
        the moment it's minted (`on_trace` observes it then — how the dashboard watches
        live) and captured the moment its run completes — so a hook that raises after
        some runs finished still yields a record containing them (the completed subset,
        in completion order; on success the hook's returned order is authoritative).
        A `rollout()`/`score()` exception is a rollout-level failure: it lands on the
        record's `errors`, never on a trace — per-agent failures are data on their
        traces, and what a failed participant means is the hook's call."""
        agents = self._agents_for(ctx)
        completed: list[Trace] = []
        handed = {
            name: _RoleAgent(
                agents[name],
                role=name if self._stamp_roles else None,
                trainable=self._roles[name].trainable,
                completed=completed,
                on_trace=on_trace,
            )
            for name in self._roles
        }
        record: RolloutRecord = RolloutRecord(
            env=self.config.env_id,
            task=TraceTask(type=type(task).__name__, data=task.data),
        )
        try:
            traces = list(await self.rollout(task, handed))
            try:
                async with asyncio.timeout(self.config.timeout.score):
                    await self.score(task, traces)
            except TimeoutError:
                raise TimeoutError(
                    f"{type(self).__name__}.score() exceeded its "
                    f"{self.config.timeout.score:g}s deadline (--timeout.score)"
                ) from None
            record.traces = traces
        except Exception as e:
            record.errors.append(
                Error(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                )
            )
            record.traces = list(completed)
        return record

    def slots(self, task: Task, n: int = 1) -> list[RunSlot]:
        """Plan `n` independent env-rollouts of `task` — one observable `RunSlot`
        each, run to a record by `run_slot`. `-r n` means exactly this: n records per
        task, nothing coupling them (env-internal multiplicity is the env's own knob).
        Harness capability (tools / user sim / container) is class-level and already
        checked per role at construction (`validate_pairing`)."""
        if n < 1:
            raise ValueError("a task needs at least one rollout (n >= 1)")
        return [RunSlot(task) for _ in range(n)]

    async def run_slot(
        self,
        slot: RunSlot,
        ctx: ModelContext,
        semaphore: asyncio.Semaphore | None = None,
        on_complete: Callable[[RolloutRecord], Awaitable[None]] | None = None,
    ) -> RolloutRecord:
        """Run one planned env-rollout to its finished record (whole-record retries
        per `retries.rollout`; the role agents resolve the task's runtime and timeouts
        per run — cli/toml > task > default, None = no limit). `slot` stays observable
        while it happens; `on_complete` fires the moment the record is final — the
        runners' persistence hook."""

        async def attempt() -> RolloutRecord:
            slot.traces = []  # a retry shows the fresh attempt's traces
            return await self.run_record(slot.task, ctx, on_trace=slot.traces.append)

        async with semaphore or contextlib.nullcontext():
            record = await run_record_with_retry(attempt, self.config.retries.rollout)
        # The record is authoritative: the hook's returned order, or (on a hook
        # failure) the completed subset.
        slot.traces = list(record.traces)
        slot.record = record
        slot.done = True
        if on_complete is not None:
            await on_complete(record)
        # hand freed per-turn request bodies (base64 images) back to the OS
        await trim_memory_periodically()
        return record

    @contextlib.asynccontextmanager
    async def serving(self):
        """Hold the env-level serving resources for the duration of an eval: the shared tool
        servers (built once, see `shared_tools`), the interception, and whatever the env's
        own `setup()` brings up. Stash them so every rollout run inside this context
        rides them (through the env's agents) — that's what keeps both eval runners
        (in-process and env-server) on one serving path. Plan and run slots inside this
        context; the resources are torn down on exit (`teardown()`, then the framework's)."""
        async with self.shared_tools() as shared:
            interception = make_interception(
                self.config.interception, requires_tunnel=self._requires_tunnel(shared)
            )
            async with interception:
                self._shared_tools = shared
                self._interception = interception
                self._agents = None  # rebuilt on the live resources (and dropped after)
                self._agents_ctx = None
                try:
                    await self.setup()
                    yield
                finally:
                    self._shared_tools = {}
                    self._interception = None
                    self._agents = None
                    self._agents_ctx = None
                    try:
                        await self.teardown()
                    finally:
                        clients, self._role_clients = self._role_clients, {}
                        for client in clients.values():
                            with contextlib.suppress(Exception):
                                await client.close()

    def _runs_local(self) -> bool:
        """Whether every role's runtime policy is local — the env-level stand-in for
        the single harness's `runtime_is_local` (any remote role means tunnels)."""
        return all(
            runtime_is_local(harness.config.runtime)
            for harness in self._harnesses.values()
        )

    def _requires_tunnel(self, shared: dict[str, SharedToolServer]) -> bool:
        """`requires_tunnel` over the consumers known before any rollout: the role
        runtimes (by config), the live `shared` servers, and the task class's tool
        servers — read class-level (like `validate_pairing`) with their configs resolved
        the way `Task.server_config` resolves them. A task that *overrides* that pairing
        isn't statically knowable, so it conservatively counts as remote (the tunnel then
        reaches everything; a wrongly-assumed localhost would reach nothing remote)."""
        task_cls = generic_type(type(self.taskset), Task, origin=Taskset) or Task
        server_classes = [*task_cls.tools]
        if server_classes and task_cls.server_config is not Task.server_config:
            return True
        sole = len({*task_cls.tools}) == 1
        configs = [
            resolve_server_config(
                task_cls.__name__, self.taskset.config.task, server_cls, sole=sole
            )
            for server_cls in server_classes
        ]
        return requires_tunnel(self._runs_local(), configs, shared.values())

    @contextlib.asynccontextmanager
    async def shared_tools(self):
        servers = self.taskset.tool_servers()
        if not servers:
            yield {}
            return
        async with serve_shared(servers, harness_is_local=self._runs_local()) as shared:
            yield shared


class _RoleAgent:
    """A role's `Agent` as handed to `Environment.rollout`: every trace it produces is
    stamped (`role`/`trainable`) the moment it's minted and captured in `completed` the
    moment its run finishes — the crash-safe record source when the hook then raises.
    Everything else delegates to the wrapped agent."""

    def __init__(
        self,
        agent: "Agent",
        *,
        role: str | None,
        trainable: bool,
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
    ) -> None:
        self._agent = agent
        self._role = role
        self._trainable = trainable
        self._completed = completed
        self._on_trace = on_trace

    def __getattr__(self, name: str):
        return getattr(self._agent, name)

    async def run(
        self,
        task: Task,
        *,
        runtime: "Runtime | None" = None,
        user: "Respond | None" = None,
        user_opens: bool = False,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        def watch(trace: Trace) -> None:
            trace.role = self._role
            trace.trainable = self._trainable
            if self._on_trace is not None:
                self._on_trace(trace)
            if on_trace is not None:
                on_trace(trace)

        trace = await self._agent.run(
            task, runtime=runtime, user=user, user_opens=user_opens, on_trace=watch
        )
        self._completed.append(trace)
        return trace

    def chat(self, task: Task, *, runtime: "Runtime | None" = None):
        """`Agent.chat`, but the run underneath is this wrapper's `run` — so a chat
        driven from `Environment.rollout` still stamps roles and stays crash-safe."""
        from verifiers.v1.agent import Agent

        return Agent.chat(self, task, runtime=runtime)  # type: ignore[arg-type]
