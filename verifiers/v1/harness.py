"""The harness: a program that runs in a runtime and drives the conversation.

An `Harness` provisions itself into the (already-started) runtime and runs there;
its model calls hit the interception server, which records the turns. Concrete
harnesses differ only in how they provision + invoke — `DefaultHarness` stages a small
script and runs `python3`; `RLMHarness` installs the rlm binary and runs it. The
runtime and the interception server are owned by the Rollout.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar, Generic, TypeVar

from pydantic_config import BaseConfig

from verifiers.v1.clients import RolloutContext
from verifiers.v1.decorators import discover_decorated, invoke
from verifiers.v1.errors import ProgramError
from verifiers.v1.runtimes import DockerConfig, ProgramResult, Runtime, RuntimeConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class HarnessConfig(BaseConfig):
    """A harness's config — subclass per harness (each pins `id` to the harness id). Mirrors
    `TasksetConfig`: the base type names the field, the concrete subclass is resolved by id
    (no closed union)."""

    id: str = "default"
    """The harness id — the discriminator that selects this harness (built-in registry, else
    a package imported by id). Set via `--harness.id`."""
    runtime: RuntimeConfig = DockerConfig()
    """Where the harness runs (subprocess / docker / prime). Lives on the harness — it's
    the harness's box; tool servers have their own placement (see `TasksetConfig.tools`)."""


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(ABC, Generic[ConfigT]):
    """Generic over its config type, so `self.config` is fully typed in subclasses
    (e.g. `RLMHarness(Harness[RLMHarnessConfig])`)."""

    APPENDS_SYSTEM_PROMPT: ClassVar[bool] = (
        False  # emit task.system_prompt as a system message (else fold into the user message)
    )

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def resolve_prompt(self, task: Task) -> tuple[str | None, str]:
        """Resolve `(system_prompt, user_instruction)` for this harness. If the harness
        appends the system prompt natively, returns it separately; otherwise folds it into
        the user instruction (warning that it isn't sent as a system message)."""
        system = task.system_prompt
        if system is None or self.APPENDS_SYSTEM_PROMPT:
            return system if self.APPENDS_SYSTEM_PROMPT else None, task.instruction
        logger.warning(
            "Harness %r does not support a separate system prompt; prepending "
            "task.system_prompt to the user instruction.",
            self.config.id,
        )
        return None, f"{system}\n\n{task.instruction}"

    async def run(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> None:
        """Run the harness in `runtime` (via `launch`) and handle its exit; its model calls
        reach the interception server at `endpoint`, and `mcp_urls` are the task's tool
        servers (name -> URL) to expose to the model."""
        result = await self.launch(ctx, trace, runtime, endpoint, secret, mcp_urls)
        if trace.stop_condition is not None:
            return  # a @stop refused a turn mid-rollout; the harness's exit is expected
        if result.exit_code != 0:
            # The real cause is at the END of a traceback, so keep the tail.
            raise ProgramError(
                f"harness exited {result.exit_code}: {result.stderr.strip()[-2000:]}"
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
        for fn, result in zip(
            fns, await asyncio.gather(*(invoke(fn, available) for fn in fns))
        ):
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
    ) -> ProgramResult:
        """Run the harness program in `runtime` to completion and return its result. The
        task is `trace.task`; model calls should reach the interception server at
        `endpoint` (bearer token `secret`); `mcp_urls` are the task's tool servers
        (name -> URL) to wire in. Each harness owns the env its program needs — read
        `ctx.model` for the model id (the default/compact harnesses set OPENAI_*; rlm sets
        RLM_* too). The uv-script harnesses just `runtime.run_uv_script(...)`."""
