from __future__ import annotations

import asyncio
import base64
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.deployment import (
    DOCKER_HARNESS_IMAGE,
    DOCKER_HARNESS_IMAGE_PREFIX,
    HARNESS_IMAGE,
    HARNESS_IMAGE_PREFIX,
)
from verifiers.v1.configs.harness import DialectName, HarnessConfig, skill_destination
from verifiers.v1.errors import (
    MODEL_TRANSPORT_ERROR_EXIT_CODE,
    HarnessError,
    InterceptionError,
    SandboxError,
    boundary,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.types import Message, Messages, ToolCall
from verifiers.v1.utils.decorators import (
    discover_decorated,
    invoke_all,
    seed,
    unseed,
)
from verifiers.v1.utils.logging import get_logger

if TYPE_CHECKING:
    from verifiers.v1.harnesses.utils.sandbox_tools import RemoteToolsSpec

    # Annotation-only: `Trace` appears in signatures only, so this module stays
    # importable below the trace record.
    from verifiers.v1.runtimes.deployment import ExecutionConnection
    from verifiers.v1.trace import Trace

logger = get_logger(__name__)

_WARNED_SYSTEM_PROMPT: set[str] = set()
"""Harness ids already warned about folding the system prompt: the warning is per
config, so it fires once per process, not once per rollout."""


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(ABC, Generic[ConfigT]):
    HARNESS_TARGET: ClassVar[str | None] = None
    """Dockerfile stage for this harness's dedicated image (`None` = `base`)."""
    HARNESS_TAG: ClassVar[str | None] = None
    """Tag suffix when the config carries no `version`/`commit` field."""
    APPENDS_SYSTEM_PROMPT: ClassVar[bool] = False
    """Emit `TaskData.system_prompt` separately instead of folding it into the user prompt."""
    SUPPORTS_MCP: ClassVar[bool] = False
    SUPPORTS_TOOL_INTERCEPTION: ClassVar[bool] = False
    """Whether the program asks the rollout's `/tool` gate (handed over as
    `tool_interception_url`) before executing each tool call, so a request hook can
    deny a call and the model sees the policy's result in its place. Without the gate
    a pre-execution rewrite can only end the rollout."""
    SUPPORTS_RESUME: ClassVar[bool] = False
    """Whether the default `resume()` can relaunch this harness from the
    accumulated Messages transcript."""
    EXECUTES_CODE: ClassVar[bool] = True
    """Whether the program hands the model local execution in the runtime — true for
    every real harness; the tool-less chat loops (`null`) override to False. Read
    where model-directed execution changes the rules: the subprocess-on-host
    warning, the judge env's sandbox requirement."""
    SUPPORTS_SKILLS: ClassVar[bool] = False
    """Whether the program discovers SKILL.md skills — it calls
    `install_skills` with a run-scoped discovery location; configuring
    `skills` on a harness without support is rejected up front."""
    NEEDS_CONTAINER: ClassVar[bool] = True
    """Whether the program must run in a container runtime: True for every harness
    that installs and drives a third-party program — on the host (subprocess) it
    leaks host state (auth, config, processes) both ways. Only the minimal
    in-house loops (`bash`, `null`) override to False."""

    def __init__(self, config: ConfigT) -> None:
        self.config = config

    @property
    def harness_target(self) -> str:
        """Dockerfile build target for this harness's sandbox image."""
        return self.HARNESS_TARGET or "base"

    @property
    def harness_tag(self) -> str:
        """Image tag corresponding to this harness's default pinned version."""
        if self.HARNESS_TAG:
            return self.HARNESS_TAG
        default_cfg = type(self.config).model_construct()
        raw = getattr(default_cfg, "version", getattr(default_cfg, "commit", None))
        if not raw:
            return "v1"
        tag = "".join(c if c.isalnum() or c in "._-" else "-" for c in str(raw)).strip(
            ".-"
        )
        return tag[:24] if len(tag) == 40 else tag

    def harness_image(self, runtime_type: str = "prime") -> str:
        """Prepared container image for this harness's dedicated sandbox."""
        prefix = (
            DOCKER_HARNESS_IMAGE_PREFIX
            if runtime_type == "docker"
            else HARNESS_IMAGE_PREFIX
        )
        target = self.harness_target
        if target == "base":
            return DOCKER_HARNESS_IMAGE if runtime_type == "docker" else HARNESS_IMAGE
        return f"{prefix}-{target}:{self.harness_tag}"

    def replay_view(self, message: Message, dialect: DialectName | None) -> Message:
        """`message` as this harness re-sends it on later requests over `dialect` (None
        before the first intercepted request). Override only for
        deterministic cosmetic rewrites (e.g. dropped whitespace-only text), never for
        content changes, so a replayed turn matches the turn it came from in the trace
        graph instead of forking it."""
        return message

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
            raise TypeError(
                f"Harness {self.config.id!r} cannot fold a system prompt into a "
                f"{'Messages' if prompt is not None else 'None'} prompt; set "
                "APPENDS_SYSTEM_PROMPT to emit it as a system message."
            )
        if self.config.id not in _WARNED_SYSTEM_PROMPT:
            _WARNED_SYSTEM_PROMPT.add(self.config.id)
            logger.warning(
                "Harness %s does not support a separate system prompt - prepending "
                "task.system_prompt to the user prompt",
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
        """Provision this harness through `runtime` before its execution timeout starts."""

    async def remote_tools(
        self, runtime: Runtime, data: TaskData
    ) -> RemoteToolsSpec | None:
        """Prepare an adapter to run in A against B. Native workers return None and register
        their workspace commands with the connection in setup_execution()."""
        return None

    def can_gate_tool(self, call: ToolCall) -> bool:
        """Whether this native call can consume a policy replacement at its gate."""
        return self.SUPPORTS_TOOL_INTERCEPTION

    def check_tool_call(self, call: ToolCall, *, intercepted: bool) -> None:
        """Refuse native modes whose execution cannot satisfy the active policy."""

    async def setup_execution(
        self, runtime: Runtime, connection: ExecutionConnection
    ) -> None:
        """Concurrently with `setup(runtime)` in the harness sandbox, prepare native
        workers in `connection.runtime` and register them with the shared execution connection.
        The connection owns worker streams and cleanup on every runtime."""
        # Adapters bind through remote_tools(); native workers override this hook.

    async def install_skills(
        self, runtime: Runtime, dest: str, *, mirror_to: Runtime | None = None
    ) -> None:
        """Install `config.skills` in the program's discovery directory.
        Runtime roots are copied in place (or bridged from the workspace);
        host skill folders are uploaded."""
        bridge = getattr(runtime, "_execution_bridge", None)
        for skill in self.config.skills:
            target_dir = skill_destination(skill, dest)
            if isinstance(skill, dict):
                if bridge is None:
                    copied = await runtime.run(
                        [
                            "bash",
                            "-e",
                            "-c",
                            (
                                '[ -d "$1" ] || exit 0; mkdir -p -- "$2" && '
                                "cd -- \"$1\" && find . -type f -print0 | while IFS= read -r -d '' f; do "
                                'mkdir -p -- "$2/$(dirname -- "$f")" && cp -p -- "$f" "$2/$f"; '
                                "done"
                            ),
                            "vf-skills",
                            skill["runtime"],
                            target_dir,
                        ],
                        {},
                    )
                    if copied.exit_code:
                        raise RuntimeError(
                            f"installing runtime skills from {skill['runtime']!r} failed: {copied.stderr}"
                        )
                    continue
                archived = await bridge.runtime.run(
                    [
                        "bash",
                        "-e",
                        "-c",
                        (
                            '[ -d "$1" ] || exit 0; cd -- "$1" && '
                            "find . -type f -print0 | while IFS= read -r -d '' f; do "
                            'mode=0; [ -x "$f" ] && mode=1; '
                            'printf "%s\\0%s\\0%s\\0" "${f#./}" "$mode" "$(base64 < "$f" | tr -d "\\n")"; '
                            "done"
                        ),
                        "vf-skills",
                        skill["runtime"],
                    ],
                    {},
                )
                if archived.exit_code:
                    raise RuntimeError(
                        f"installing runtime skills from {skill['runtime']!r} failed: {archived.stderr}"
                    )
                uploads = []
                executables = []
                if archived.stdout:
                    parts = archived.stdout.split("\0")
                    for i in range(0, len(parts) - 1, 3):
                        rel, mode, b64 = parts[i], parts[i + 1], parts[i + 2]
                        target = f"{target_dir}/{rel}"
                        uploads.append((target, base64.b64decode(b64)))
                        if mode == "1":
                            executables.append(target)
                if mirror_to is not None and skill["runtime"] != target_dir:
                    copied_b = await mirror_to.run(
                        [
                            "bash",
                            "-e",
                            "-c",
                            (
                                '[ -d "$1" ] || exit 0; mkdir -p -- "$2" && '
                                "cd -- \"$1\" && find . -type f -print0 | while IFS= read -r -d '' f; do "
                                'mkdir -p -- "$2/$(dirname -- "$f")" && cp -p -- "$f" "$2/$f"; '
                                "done"
                            ),
                            "vf-skills",
                            skill["runtime"],
                            target_dir,
                        ],
                        {},
                    )
                    if copied_b.exit_code:
                        raise RuntimeError(
                            f"installing runtime skills from {skill['runtime']!r} failed: {copied_b.stderr}"
                        )
                targets = (runtime,)
            else:
                # Resolve so `.`/`..` entries get their real folder name (and can't
                # place files outside `dest`).
                skill = skill.resolve()
                if not skill.is_dir():
                    raise ValueError(f"skill {str(skill)!r} is not a folder")
                uploads = []
                executables = []
                for file in skill.rglob("*"):
                    if not file.is_file():
                        continue
                    target = f"{target_dir}/{file.relative_to(skill).as_posix()}"
                    uploads.append((target, file.read_bytes()))
                    if os.access(file, os.X_OK):
                        executables.append(target)
                targets = (runtime, mirror_to) if mirror_to is not None else (runtime,)
            for rt in targets:
                results = await asyncio.gather(
                    *(rt.write(target, data) for target, data in uploads),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                if executables:
                    await rt.run(["chmod", "+x", *executables], {})

    async def _check_result(
        self, trace: Trace, runtime: Runtime, result: ProgramResult
    ) -> None:
        if trace.stop_condition is not None:
            return  # a @stop refused a turn mid-rollout; the exit is expected
        if result.exit_code == 0:
            return
        # The real cause is at the END of a traceback, so keep the tail.
        detail = (result.stderr or result.stdout).strip()[-2000:] or "<no output>"
        if not await runtime.alive():
            raise SandboxError(
                f"runtime died under harness {self.config.id!r} "
                f"(exit {result.exit_code}): {detail}"
            )
        if result.exit_code == MODEL_TRANSPORT_ERROR_EXIT_CODE:
            raise InterceptionError(
                f"harness {self.config.id!r} model connection to interception failed: {detail}"
            )
        raise HarnessError(
            f"harness {self.config.id!r} exited {result.exit_code}: {detail}"
        )

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> HarnessSession:
        """Create the rollout-scoped handle that drives this harness.

        The default adapts the existing launch/resume contract. Stateful harness
        transports override this factory so one handle can own their live process,
        connection, or native session for the rollout's full interaction.
        """
        return HarnessSession(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            tool_interception_url,
        )

    async def score(self, trace: Trace, runtime: Runtime) -> None:
        """Run this harness's `@metric` methods over the finished trace, recording
        each into `trace.metrics`. Metrics declare what they need (`task`, `trace`,
        `runtime`) and can read what the harness left behind in the runtime."""
        available = {
            "task": trace.task.data,
            "trace": trace,
            "runtime": runtime,
        }
        fns = discover_decorated(self, "metric")
        seed(trace.metrics, (fn.__name__ for fn in fns))
        async with boundary(HarnessError, f"harness {self.config.id!r} metric"):
            results = await invoke_all(fns, available)
        for fn, result in zip(fns, results):
            if isinstance(result, Mapping):
                unseed(trace.metrics, fn.__name__)
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
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        """Run the next segment of an exchange this trace already carries: the user
        spoke (`messages`), the program answers — with the whole conversation behind
        it. The default relaunches the program on the accreted conversation (the
        trace's current branch plus `messages`) as a Messages prompt: correct for any
        stateless chat program, including the very first segment of an exchange the
        user opens (an empty branch). A harness with its own session state overrides
        this or returns a specialized session handle with a native continuation
        instead of replaying a conversation it already owns."""
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
        kwargs = (
            {"tool_interception_url": tool_interception_url}
            if self.SUPPORTS_TOOL_INTERCEPTION
            else {}
        )
        return await self.launch(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data.model_copy(update={"prompt": conversation}),
            **kwargs,
        )

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
        """Run the harness program through `runtime` to completion and return its result.
        `data` is this segment's wire view of the task (`resolve_prompt(data)` — for
        a resumed exchange its prompt is the accreted conversation, and it may
        differ from `trace.task.data`, the run's recorded view); model calls must
        reach the interception server at `endpoint` (bearer token `secret`);
        `mcp_urls` are the task's tool servers to wire in. Each harness owns the
        env its program needs (the compact harness sets OPENAI_*).

        The interception is the contract, not the process: a harness may run its
        loop in-process instead of launching a program, as long as every model call
        goes through `endpoint` + `secret` — it then returns a synthetic success
        `ProgramResult`, and the trace is the record of what ran."""


class HarnessSession:
    """One rollout's stateful handle onto one harness execution.

    The base adapter preserves the segment-oriented harness interface by invoking
    `launch()` once and `resume()` for later caller turns. Specialized handles can
    override `_run()` and `close()` to retain transport state across those turns.
    """

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.trace = trace
        self.runtime = runtime
        self.endpoint = endpoint
        self.secret = secret
        self.mcp_urls = mcp_urls
        self.data = data
        self.tool_interception_url = tool_interception_url
        self._closed = False

    async def steer(self, message: str, *, message_id: str | None = None) -> dict:
        """Deliver a user message during a running turn, if supported by the harness."""
        raise NotImplementedError(
            f"harness {self.harness.config.id!r} does not support steering"
        )

    async def turn(self, messages: Messages | None = None) -> None:
        """Run one harness segment while retaining session state for the next."""
        if self._closed:
            raise HarnessError(
                f"harness {self.harness.config.id!r} session is already closed"
            )
        async with boundary(HarnessError, f"harness {self.harness.config.id!r}"):
            result = await self._run(messages)
        await self.harness._check_result(self.trace, self.runtime, result)

    async def _run(self, messages: Messages | None) -> ProgramResult:
        if messages is None:
            kwargs = (
                {"tool_interception_url": self.tool_interception_url}
                if self.harness.SUPPORTS_TOOL_INTERCEPTION
                else {}
            )
            return await self.harness.launch(
                self.ctx,
                self.trace,
                self.runtime,
                self.endpoint,
                self.secret,
                self.mcp_urls,
                self.data,
                **kwargs,
            )
        return await self.harness.resume(
            self.ctx,
            self.trace,
            self.runtime,
            self.endpoint,
            self.secret,
            self.mcp_urls,
            self.data,
            messages,
            self.tool_interception_url,
        )

    async def close(self) -> None:
        """Close session-owned resources. Idempotent."""
        self._closed = True
