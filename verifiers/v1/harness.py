from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import ClientConfig, ModelContext
from verifiers.v1.decorators import discover_decorated, invoke_all
from verifiers.v1.errors import HarnessError, boundary
from verifiers.v1.retries import RetryConfig
from verifiers.v1.utils.install import env_name
from verifiers.v1.runtimes import (
    ProgramResult,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.task import TaskData
from verifiers.v1.types import ID, Messages, SamplingConfig

if TYPE_CHECKING:
    # Annotation-only: the runtime import goes trace -> harness (AgentInfo embeds
    # AgentConfig), not the other way around.
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class TimeoutConfig(BaseConfig):
    """Per-agent wall-clock timeouts per rollout stage, in seconds (None = no
    limit); each stage falls back to the task's own `TaskTimeout` when unset."""

    setup: float | None = None  # one shared budget: task setup + provisioning
    rollout: float | None = None
    finalize: float | None = None
    scoring: float | None = None


class AgentConfig(BaseConfig):
    """One env agent: who plays it, flat — the harness (by id) with its knobs
    directly on the seat, and the agent's per-run caps. A harness declares its
    extra knobs by subclassing `AgentConfig`; the seat's config narrows to that
    subclass by `harness`, so harness fields address flat (`--env.<role>.<knob>`).
    It pins only what makes it a different actor; everything unpinned falls back —
    the model context to the run's own, the harness to the taskset's default."""

    harness: ID | None = None
    """The harness program — local package or Hub `org/name[@version]`, set through
    the seat's `--env.<role>.harness` (`--env.agent.harness` on the single-agent
    env). None = the taskset's default harness."""
    runtime: RuntimeConfig = SubprocessConfig()
    """Runtime for the harness program; tool servers choose their placement separately."""
    env: dict[str, str] = Field(default_factory=dict)
    """Extra program variables; harness-owned variables take precedence."""
    forward_env: list[str] = Field(default_factory=list)
    """Host variables to forward without writing secrets into config; explicit `env` wins."""
    disabled_tools: list[str] | None = None
    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: SamplingConfig | None = None
    """Sampling override (None = the run's sampling)."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    """Whole-run retries: rerun this agent's rollout while its trace ends with a
    retryable error (never into a borrowed box)."""
    max_turns: int | None = None
    """Max model turns per run (None = no limit). Framework-enforced (the
    interception server refuses turns past it), so it applies to any harness."""
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    """Token caps per run (None = no limit); framework-enforced between turns."""

    @model_validator(mode="before")
    @classmethod
    def _flat_harness(cls, data):
        """Point nested harness data (the old shape) at the flat surface."""
        if isinstance(data, dict) and isinstance(data.get("harness"), dict):
            raise ValueError(
                "the harness config is flat on the agent: set `harness` to the id "
                "(--env.agent.harness bash) and its knobs directly on the seat "
                "(--env.agent.runtime.type docker; TOML [env.agent.runtime])"
            )
        return data

    @property
    def name(self) -> str:
        return env_name(self.harness) if self.harness else "?"

    @property
    def resolved_env(self) -> dict[str, str]:
        forwarded = {k: os.environ[k] for k in self.forward_env if k in os.environ}
        return {**forwarded, **self.env}


ConfigT = TypeVar("ConfigT", bound=AgentConfig)


class Harness(ABC, Generic[ConfigT]):
    APPENDS_SYSTEM_PROMPT: ClassVar[bool] = False
    """Emit `TaskData.system_prompt` separately instead of folding it into the user prompt."""
    SUPPORTS_MCP: ClassVar[bool] = False
    SUPPORTS_USER_SIM: ClassVar[bool] = False
    SUPPORTS_MESSAGE_PROMPT: ClassVar[bool] = False
    EXECUTES_CODE: ClassVar[bool] = True
    """Whether the program hands the model local execution in the runtime — true for
    every real harness; the tool-less chat loops (`null`) override to False. Read
    where model-directed execution changes the rules: the subprocess-on-host
    warning, the judge env's sandbox requirement."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def resolve_prompt(
        self, task: TaskData
    ) -> tuple[str | None, str | Messages | None]:
        prompt = task.prompt
        if (
            prompt is not None
            and not isinstance(prompt, str)
            and not self.SUPPORTS_MESSAGE_PROMPT
        ):
            raise ValueError(
                f"Harness {self.config.harness!r} does not support a Messages prompt; "
                "task.prompt must be a string or None."
            )
        system = task.system_prompt
        if system is None or self.APPENDS_SYSTEM_PROMPT:
            return system if self.APPENDS_SYSTEM_PROMPT else None, prompt
        if not isinstance(prompt, str):
            raise ValueError(
                f"Harness {self.config.harness!r} cannot fold a system prompt into a "
                f"{'Messages' if prompt is not None else 'None'} prompt; set "
                "APPENDS_SYSTEM_PROMPT to emit it as a system message."
            )
        logger.warning(
            "Harness %r does not support a separate system prompt; prepending "
            "task.system_prompt to the user prompt.",
            self.config.harness,
        )
        return None, f"{system}\n\n{prompt}"

    async def setup(self, runtime: Runtime) -> None:
        """Provision this harness in `runtime` before its execution timeout starts."""

    async def run(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> None:
        async with boundary(HarnessError, f"harness {self.config.harness!r}"):
            result = await self.launch(ctx, trace, runtime, endpoint, secret, mcp_urls)
        if trace.stop_condition is not None:
            return  # a @stop refused a turn mid-rollout; the harness's exit is expected
        if result.exit_code != 0:
            # The real cause is at the END of a traceback, so keep the tail.
            detail = (result.stderr or result.stdout).strip()[-2000:] or "<no output>"
            raise HarnessError(
                f"harness {self.config.harness!r} exited {result.exit_code}: {detail}"
            )
        trace.stop("agent_completed")

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        """Run this harness's `@metric` methods over the finished trace, recording
        each into `trace.metrics`. Metrics declare what they need (`task`, `trace`,
        `runtime`) and can read what the harness left behind in the runtime."""
        available = {"task": trace.task.data, "trace": trace, "runtime": runtime}
        fns = discover_decorated(self, "metric")
        async with boundary(HarnessError, f"harness {self.config.harness!r} metric"):
            results = await invoke_all(fns, available)
        for fn, result in zip(fns, results):
            if isinstance(result, Mapping):
                trace.record_metrics(result)
            else:
                trace.record_metric(fn.__name__, result)

    @abstractmethod
    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        """Run the harness program in `runtime` to completion and return its result.
        The task is `trace.task.data`; model calls must reach the interception
        server at `endpoint` (bearer token `secret`); `mcp_urls` are the task's tool
        servers to wire in. Each harness owns the env its program needs (the
        bash/compact harnesses set OPENAI_*).

        The interception is the contract, not the process: a harness may run its
        loop in-process instead of launching a program, as long as every model call
        goes through `endpoint` + `secret` — it then returns a synthetic success
        `ProgramResult`, and the trace is the record of what ran."""
