import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Generic, TypeVar

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke_all, reject_agent_scope
from verifiers.v1.errors import HarnessError, boundary
from verifiers.v1.utils.install import env_name
from verifiers.v1.runtimes import (
    ProgramResult,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import ID, Messages

logger = logging.getLogger(__name__)


class HarnessConfig(BaseConfig):
    id: ID = "default"
    """Local package or Hub `org/name[@version]`, set through `--harness.id`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """Runtime for the harness program; tool servers choose their placement separately."""
    env: dict[str, str] = Field(default_factory=dict)
    """Extra program variables; harness-owned variables take precedence."""
    forward_env: list[str] = Field(default_factory=list)
    """Host variables to forward without writing secrets into config; explicit `env` wins."""
    disabled_tools: list[str] | None = None

    @property
    def name(self) -> str:
        return env_name(self.id)

    @property
    def resolved_env(self) -> dict[str, str]:
        forwarded = {k: os.environ[k] for k in self.forward_env if k in os.environ}
        return {**forwarded, **self.env}


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(ABC, Generic[ConfigT]):
    APPENDS_SYSTEM_PROMPT: ClassVar[bool] = False
    """Emit `TaskData.system_prompt` separately instead of folding it into the user prompt."""
    SUPPORTS_MCP: ClassVar[bool] = False
    SUPPORTS_USER_SIM: ClassVar[bool] = False
    SUPPORTS_MESSAGE_PROMPT: ClassVar[bool] = False

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
                f"Harness {self.config.id!r} does not support a Messages prompt; "
                "task.prompt must be a string or None."
            )
        system = task.system_prompt
        if system is None or self.APPENDS_SYSTEM_PROMPT:
            return system if self.APPENDS_SYSTEM_PROMPT else None, prompt
        if not isinstance(prompt, str):
            raise ValueError(
                f"Harness {self.config.id!r} cannot fold a system prompt into a "
                f"{'Messages' if prompt is not None else 'None'} prompt; set "
                "APPENDS_SYSTEM_PROMPT to emit it as a system message."
            )
        logger.warning(
            "Harness %r does not support a separate system prompt; prepending "
            "task.system_prompt to the user prompt.",
            self.config.id,
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
        async with boundary(HarnessError, f"harness {self.config.id!r}"):
            result = await self.launch(ctx, trace, runtime, endpoint, secret, mcp_urls)
        if trace.stop_condition is not None:
            return  # a @stop refused a turn mid-rollout; the harness's exit is expected
        if result.exit_code != 0:
            # The real cause is at the END of a traceback, so keep the tail.
            detail = (result.stderr or result.stdout).strip()[-2000:] or "<no output>"
            raise HarnessError(
                f"harness {self.config.id!r} exited {result.exit_code}: {detail}"
            )
        trace.stop("agent_completed")

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        """Run this harness's `@metric` methods over the finished trace, concurrently,
        recording each into `trace.metrics`. Mirrors `Task.score` (which the
        Rollout runs in parallel with this); metrics declare what they need (`task`,
        `trace`, `runtime`) and can read what the harness left behind in the runtime.
        No-op for an harness with no `@metric`s."""
        available = {"task": trace.task.data, "trace": trace, "runtime": runtime}
        fns = discover_decorated(self, "metric")
        # Before the HarnessError boundary: mis-scoped agent= is an authoring error.
        reject_agent_scope(fns, owner=f"harness {self.config.id!r}")
        async with boundary(HarnessError, f"harness {self.config.id!r} metric"):
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
        """Run the harness program in `runtime` to completion and return its result. The
        task is `trace.task.data`; model calls should reach the interception server at
        `endpoint` (bearer token `secret`); `mcp_urls` are the task's tool servers
        (name -> URL) to wire in. Each harness owns the env its program needs — read
        `ctx.model` for the model id (the default/compact harnesses set OPENAI_*; rlm sets
        RLM_* too). UV-script harnesses prepare dependencies in `setup`, then launch the
        returned argv through `runtime.run_program(...)` here."""
