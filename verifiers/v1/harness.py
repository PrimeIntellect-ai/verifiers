"""The harness: a program that runs in a runtime and drives the conversation.

A `Harness` provisions itself into the already-started runtime, then runs there;
its model calls hit the interception server, which records the turns. Provisioning
runs during rollout setup; only execution counts against the harness timeout. The
runtime and the interception server are owned by the Rollout.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Generic, TypeVar

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.clients import RolloutContext
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import HarnessError, boundary
from verifiers.v1.utils.install import env_name
from verifiers.v1.runtimes import (
    ProgramResult,
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import EnvId, Messages

logger = logging.getLogger(__name__)


class HarnessConfig(BaseConfig):
    """A harness's config — subclass per harness to add run knobs. Mirrors `TasksetConfig`: the
    base type names the field, the concrete subclass is resolved by id (no closed union) — the
    id is supplied by the caller (`--harness.id` / toml / a taskset's bundled harness), never
    pinned on the subclass."""

    id: EnvId = "default"
    """The harness id, which selects this harness: a local package, or an
    `org/name[@version]` package installed on demand from the Environments Hub (see
    `EnvId`). Set via `--harness.id`."""
    runtime: RuntimeConfig = SubprocessConfig()
    """Where the harness runs (subprocess / docker / prime). Subprocess by default — a local
    process on the host; a taskset that needs a container (its own image, or NEEDS_CONTAINER)
    selects `--harness.runtime.type docker` (or prime/modal). Lives on the harness — it's the
    harness's box; tool servers have their own placement (see `TasksetConfig.tools`)."""
    env: dict[str, str] = Field(default_factory=dict)
    """Additional environment variables for the harness program. Harness-owned endpoint,
    authentication, and model variables take precedence."""
    disabled_tools: list[str] | None = None
    """Harness-specific tool names to disable."""

    @property
    def name(self) -> str:
        """The harness's package name (the id with any org / version stripped)."""
        return env_name(self.id)


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(ABC, Generic[ConfigT]):
    """Generic over its config type, so `self.config` is fully typed in subclasses
    (e.g. `RLMHarness(Harness[RLMHarnessConfig])`)."""

    APPENDS_SYSTEM_PROMPT: ClassVar[bool] = False
    """Emit task.system_prompt as a system message (else fold into the user message)."""
    SUPPORTS_MCP: ClassVar[bool] = True
    """Expose a task's MCP tool servers to the model; set False for harnesses without an MCP client."""
    SUPPORTS_USER_SIM: ClassVar[bool] = False
    """Drive a task's user simulator (multi-turn user injection); opt in per harness."""
    SUPPORTS_MESSAGE_PROMPT: ClassVar[bool] = False
    """Accept a Messages-list task.prompt (e.g. an image-bearing prompt); opt in per harness."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def resolve_prompt(self, task: Task) -> tuple[str | None, str | Messages | None]:
        """Resolve `(system_prompt, prompt)` for this harness. If the harness
        appends the system prompt natively, returns it separately; otherwise folds it into
        the user prompt (warning that it isn't sent as a system message). A `Messages`
        prompt (e.g. an image-bearing prompt) is only allowed for harnesses that set
        `SUPPORTS_MESSAGE_PROMPT`. A `None` prompt means the task has no prompt —
        the user simulator opens the conversation (see `Taskset.user`); the harness emits no
        opening user message."""
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
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        user_url: str | None = None,
    ) -> None:
        """Run the harness in `runtime` (via `launch`) and handle its exit; its model calls
        reach the interception server at `endpoint`, and `mcp_urls` are the task's tool
        servers (name -> URL) to expose to the model. `user_url` is the interception server's
        `/user` endpoint when the task has a user simulator and this harness supports driving one
        (`SUPPORTS_USER_SIM`), else None — a supporting harness POSTs each no-tool-call turn there."""
        async with boundary(HarnessError, f"harness {self.config.id!r}"):
            result = await self.launch(
                ctx, trace, runtime, endpoint, secret, mcp_urls, user_url
            )
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
        recording each into `trace.metrics`. Mirrors `Taskset.score` (which the
        Rollout runs in parallel with this); metrics declare what they need (`task`,
        `trace`, `runtime`) and can read what the harness left behind in the runtime.
        No-op for an harness with no `@metric`s."""
        available = {"task": trace.task, "trace": trace, "runtime": runtime}
        fns = discover_decorated(self, "metric")
        async with boundary(HarnessError, f"harness {self.config.id!r} metric"):
            results = (
                [await invoke(fn, available) for fn in fns]
                if len(fns) < 2
                else await asyncio.gather(*(invoke(fn, available) for fn in fns))
            )
        for fn, result in zip(fns, results):
            if isinstance(result, Mapping):
                trace.record_metrics(result)
            else:
                trace.record_metric(fn.__name__, result)

    @abstractmethod
    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        user_url: str | None = None,
    ) -> ProgramResult:
        """Run the harness program in `runtime` to completion and return its result. The
        task is `trace.task`; model calls should reach the interception server at
        `endpoint` (bearer token `secret`); `mcp_urls` are the task's tool servers
        (name -> URL) to wire in. `user_url` is the `/user` endpoint to drive the task's user
        simulator over (only set for harnesses that set `SUPPORTS_USER_SIM`; None otherwise) —
        POST `{"message": <assistant text>}` with the same bearer `secret` on each no-tool-call
        turn, append the returned `messages`, and re-prompt until `done`. Each harness owns the env
        its program needs — read `ctx.model` for the model id (the default/compact harnesses set
        OPENAI_*; rlm sets RLM_* too). UV-script harnesses prepare dependencies in `setup`, then
        launch the returned argv through `runtime.run_program(...)` here."""
