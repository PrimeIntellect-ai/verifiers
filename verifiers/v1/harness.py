from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar


from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import discover_decorated, invoke_all
from verifiers.v1.errors import HarnessError, boundary
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.types import Messages

if TYPE_CHECKING:
    # Annotation-only: `Trace` appears in signatures only, so this module stays
    # importable below the trace record.
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(ABC, Generic[ConfigT]):
    APPENDS_SYSTEM_PROMPT: ClassVar[bool] = False
    """Emit `TaskData.system_prompt` separately instead of folding it into the user prompt."""
    SUPPORTS_MCP: ClassVar[bool] = False
    SUPPORTS_RESUME: ClassVar[bool] = False
    """Whether the default `resume()` can relaunch this harness from the
    accumulated Messages transcript."""
    EXECUTES_CODE: ClassVar[bool] = True
    """Whether the program hands the model local execution in the runtime — true for
    every real harness; the tool-less chat loops (`null`) override to False. Read
    where model-directed execution changes the rules: the subprocess-on-host
    warning, the judge env's sandbox requirement."""
    SUPPORTS_SKILLS: ClassVar[bool] = False
    """Whether the program discovers SKILL.md skills — its `setup` calls
    `install_skills` with the program's fixed discovery location; configuring
    `skills` on a harness without support is rejected up front."""
    NEEDS_CONTAINER: ClassVar[bool] = True
    """Whether the program must run in a container runtime: True for every harness
    that installs and drives a third-party program — on the host (subprocess) it
    leaks host state (auth, config, processes) both ways. Only the minimal
    in-house loops (`bash`, `null`) override to False."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    def resolve_prompt(
        self, task: TaskData
    ) -> tuple[str | None, str | Messages | None]:
        """Resolve system-prompt placement without constraining the initial prompt.

        Each `launch()` owns the prompt shapes its program accepts. In particular,
        accepting initial Messages is distinct from `SUPPORTS_RESUME`, which means
        the default `resume()` can replay an accumulated conversation.
        """
        prompt = task.prompt
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

    def resolve_text_prompt(self, task: TaskData) -> tuple[str | None, str | None]:
        """Resolve a harness prompt whose launch program accepts only plain text."""
        system, prompt = self.resolve_prompt(task)
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(
                f"Harness {self.config.id!r} does not support a Messages prompt; "
                "task.prompt must be a string or None."
            )
        return system, prompt

    async def setup(self, runtime: Runtime) -> None:
        """Provision this harness in `runtime` before its execution timeout starts."""

    async def install_skills(self, runtime: Runtime, dest: str) -> None:
        """Upload each `config.skills` folder into `runtime` at `dest/<folder name>` —
        the program's fixed skill discovery location, which a supporting harness's
        `setup` passes."""
        for skill in self.config.skills:
            # Resolve so `.`/`..` entries get their real folder name (and can't
            # place files outside `dest`).
            skill = skill.resolve()
            if not skill.is_dir():
                raise ValueError(f"skill {str(skill)!r} is not a folder")
            executables = []
            for file in sorted(skill.rglob("*")):
                if not file.is_file():
                    continue
                target = f"{dest}/{skill.name}/{file.relative_to(skill).as_posix()}"
                await runtime.write(target, file.read_bytes())
                if os.access(file, os.X_OK):
                    executables.append(target)
            if executables:
                # `write` moves bytes, not modes; restore the execute bits scripts need.
                await runtime.run(["chmod", "+x", *executables], {})

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
        if not self.SUPPORTS_RESUME:
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
