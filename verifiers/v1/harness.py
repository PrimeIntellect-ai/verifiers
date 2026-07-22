from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from pydantic import Field
from pydantic_config import BaseConfig

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke_all
from verifiers.v1.errors import HarnessError, boundary
from verifiers.v1.skills import load_skills, skills_prompt
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
    skills: list[Path] = Field(default_factory=list)
    """Agent skills to install into the runtime: each entry is one skill (a directory
    with a `SKILL.md` manifest, or the manifest itself) or a directory of such skill
    directories. Programs without native skill discovery are pointed at the installed
    skills through the prompt, so they need file access to use them."""

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
    EXECUTES_CODE: ClassVar[bool] = True
    """Whether the program hands the model local execution in the runtime — true for
    every real harness; the tool-less chat loops (`null`) override to False. Read
    where model-directed execution changes the rules: the subprocess-on-host
    warning, the judge env's sandbox requirement."""
    SKILLS_DIR: ClassVar[str] = ".vf-skills"
    """Runtime path (workspace-relative) `config.skills` are installed under."""
    DISCOVERS_SKILLS: ClassVar[bool] = False
    """Whether the program discovers skills in `SKILLS_DIR` natively; when False the
    installed skills are announced through the prompt instead."""

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
        skills = self.skills_prompt()
        if skills:
            system = f"{system}\n\n{skills}" if system else skills
        if system is None or self.APPENDS_SYSTEM_PROMPT:
            return system if self.APPENDS_SYSTEM_PROMPT else None, prompt
        if not isinstance(prompt, str):
            raise ValueError(
                f"Harness {self.config.id!r} cannot fold a system prompt into a "
                f"{'Messages' if prompt is not None else 'None'} prompt; set "
                "APPENDS_SYSTEM_PROMPT to emit it as a system message."
            )
        if task.system_prompt is not None:
            logger.warning(
                "Harness %r does not support a separate system prompt; prepending "
                "task.system_prompt to the user prompt.",
                self.config.id,
            )
        return None, f"{system}\n\n{prompt}"

    async def setup(self, runtime: Runtime) -> None:
        """Provision this harness in `runtime` before its execution timeout starts."""

    def skills_prompt(self) -> str | None:
        """The prompt section announcing `config.skills`, `None` when there are none
        or the program discovers them in `SKILLS_DIR` natively."""
        if not self.config.skills or self.DISCOVERS_SKILLS:
            return None
        return skills_prompt(load_skills(self.config.skills), self.SKILLS_DIR)

    async def install_skills(self, runtime: Runtime) -> None:
        """Write `config.skills` into `runtime` under `SKILLS_DIR`; framework-called
        alongside `setup`."""
        for skill in load_skills(self.config.skills):
            for file in skill.files():
                path = file.relative_to(skill.root).as_posix()
                await runtime.write(
                    f"{self.SKILLS_DIR}/{skill.name}/{path}", file.read_bytes()
                )

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
