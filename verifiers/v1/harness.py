from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke_all
from verifiers.v1.errors import HarnessError, boundary
from verifiers.v1.utils.install import env_name
from verifiers.v1.runtimes import (
    ProgramResult,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.task import TaskData
from verifiers.v1.types import ID, Messages

if TYPE_CHECKING:
    # Annotation-only: the runtime import goes trace -> harness (AgentInfo embeds
    # HarnessConfig), not the other way around.
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class HarnessConfig(BaseConfig):
    id: ID = "bash"
    """Local package or Hub `org/name[@version]`, set through the seat's
    `--env.<role>.harness.id` (`--env.agent.harness.id` on the single-agent env)."""
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
        data: TaskData,
        messages: Messages | None = None,
    ) -> None:
        """Run ONE segment of the exchange: the program from launch (or, with
        `messages`, the user's next turn(s) via `resume`) until it yields — a segment
        ends when the program exits. The rollout loop owns the exchange across
        segments (and stamps its end); a harness only ever sees one segment."""
        async with boundary(HarnessError, f"harness {self.config.id!r}"):
            if messages is None:
                result = await self.launch(
                    ctx, trace, runtime, endpoint, secret, mcp_urls, data
                )
            else:
                result = await self.resume(
                    ctx, trace, runtime, endpoint, secret, mcp_urls, data, messages
                )
        if trace.stop_condition is not None:
            return  # a @stop refused a turn mid-rollout; the harness's exit is expected
        if result.exit_code != 0:
            # The real cause is at the END of a traceback, so keep the tail.
            detail = (result.stderr or result.stdout).strip()[-2000:] or "<no output>"
            raise HarnessError(
                f"harness {self.config.id!r} exited {result.exit_code}: {detail}"
            )

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        """Run this harness's `@metric` methods over the finished trace, recording
        each into `trace.metrics`. Metrics declare what they need (`task`, `trace`,
        `runtime`) and can read what the harness left behind in the runtime."""
        available = {"task": trace.task.data, "trace": trace, "runtime": runtime}
        fns = discover_decorated(self, "metric")
        async with boundary(HarnessError, f"harness {self.config.id!r} metric"):
            results = await invoke_all(fns, available)
        for fn, result in zip(fns, results):
            if isinstance(result, Mapping):
                trace.record_metrics(result)
            else:
                trace.record_metric(fn.__name__, result)

    async def resume(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        messages: Messages,
    ) -> ProgramResult:
        """Run the next segment of an exchange this trace already carries: the user
        spoke (`messages`), the program answers — with the whole conversation behind
        it. The default relaunches the program on the accreted conversation (the
        trace's current branch plus `messages`) as a Messages prompt: correct for any
        stateless chat program, including the very first segment of an exchange the
        user opens (an empty branch). A harness with its own session state overrides
        this with a native continuation (codex: `codex exec resume`) instead of
        replaying a conversation it already owns."""
        if not self.SUPPORTS_MESSAGE_PROMPT:
            raise HarnessError(
                f"harness {self.config.id!r} cannot continue an exchange: it neither "
                "overrides resume() nor supports a Messages prompt for the default "
                "relaunch-on-the-conversation."
            )
        branch = trace.branches[-1].messages if trace.branches else []
        # `resolve_prompt` re-emits `data.system_prompt`; only de-duplicate those
        # system messages when it has something to re-emit. Explicit system
        # messages in a Messages prompt must survive a resumed segment.
        conversation = [
            *(m for m in branch if m.role != "system" or data.system_prompt is None),
            *messages,
        ]
        return await self.launch(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data.model_copy(update={"prompt": conversation}),
        )

    async def cleanup(self, trace: Trace, runtime: Runtime) -> None:
        """Remove harness-owned per-rollout state after scoring and before the
        runtime is released. Implementations should be idempotent."""

    @abstractmethod
    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        """Run the harness program in `runtime` to completion and return its result.
        `data` is this segment's wire view of the task (`resolve_prompt(data)` — for
        a resumed exchange its prompt is the accreted conversation, and it may
        differ from `trace.task.data`, the run's recorded view); model calls must
        reach the interception server at `endpoint` (bearer token `secret`);
        `mcp_urls` are the task's tool servers to wire in. Each harness owns the
        env its program needs (the bash/compact harnesses set OPENAI_*).

        The interception is the contract, not the process: a harness may run its
        loop in-process instead of launching a program, as long as every model call
        goes through `endpoint` + `secret` — it then returns a synthetic success
        `ProgramResult`, and the trace is the record of what ran."""
