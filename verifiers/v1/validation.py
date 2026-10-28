"""Model-free validation through the ordinary rollout and scoring lifecycle."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from verifiers.v1.configs.agent import AgentConfig, TimeoutConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.errors import TaskError
from verifiers.v1.graph import MessageNode
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import PrimeConfig, ProgramResult, Runtime, RuntimeConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, UserMessage
from verifiers.v1.utils.staging import stage_file

if TYPE_CHECKING:
    from verifiers.v1.agent import Agent

__all__ = ["ValidationAttempt", "ValidationHarness", "ValidationResult"]

GOLD_REPLY = "reply.md"
"""A gold solution's final reply: the file it leaves at `reply.md` in its solution
directory (shipped there, or written by its command) becomes the gold run's final
assistant message, so graders that read a solver's reply (a Harbor task's
`/logs/agent/final_reply.md`) see the gold one. Only a gold run reads it: a solver's
run never looks there."""
MAX_GOLD_REPLY_BYTES = 1024 * 1024

ValidationRunner = Callable[["Agent", Task, Callable[[Trace], None]], Awaitable[Trace]]
"""Optional per-attempt execution policy, returning the native graded trace."""


class ValidationAttempt(BaseModel):
    enabled: bool = True
    reward: float | None = None
    valid: bool | None = None
    trace: dict | None = None
    error: str | None = None
    infrastructure: bool = False


class ValidationResult(BaseModel):
    baseline: ValidationAttempt
    gold: ValidationAttempt

    @property
    def valid(self) -> bool | None:
        attempts = [
            attempt for attempt in (self.baseline, self.gold) if attempt.enabled
        ]
        if not attempts:
            return None
        if any(attempt.valid is None for attempt in attempts):
            return None
        return all(attempt.valid for attempt in attempts)


class ValidationHarnessConfig(HarnessConfig):
    id: str = "validation"
    source: Path | None = None
    target_dir: str = "/solution"
    command: list[str] = Field(default_factory=list)


class ValidationHarness(Harness[ValidationHarnessConfig]):
    NEEDS_CONTAINER = False
    SUPPORTS_MCP = True
    SUPPORTS_TOOL_INTERCEPTION = True
    task: Task
    kind: str

    async def launch(
        self,
        ctx,
        trace,
        runtime,
        endpoint,
        secret,
        mcp_urls,
        data,
        tool_interception_url=None,
    ):
        from verifiers.v1.utils.decorators import invoke

        # The reference is staged as root and runs as the agent would (Harbor's oracle
        # executes solve.sh as the task's [agent] user), so a solution that only works
        # as root, or a grader that checks who ran it, sees what a solver's run leaves.
        solver = runtime
        runtime = runtime.with_user(None)
        if self.config.source is not None:
            modes: list[str] = []
            for source in [self.config.source, *sorted(self.config.source.rglob("*"))]:
                relative = source.relative_to(self.config.source).as_posix()
                target = str(PurePosixPath(self.config.target_dir, relative))
                if source.is_file():
                    await stage_file(runtime, target, source)
                elif not source.is_dir():
                    continue
                # Modes as `docker cp` of the directory leaves them (git's: no group
                # or other write), the target directory's own included: an image that
                # keeps /solution root-only (0700) still lets the agent run solve.sh.
                modes += [f"{source.stat().st_mode & 0o7755:o}", target]
            for start in range(0, len(modes), 400):
                await runtime.run(
                    [
                        "sh",
                        "-c",
                        'while [ "$#" -gt 1 ]; do chmod "$1" "$2"; shift 2; done',
                        "vf",
                        *modes[start : start + 400],
                    ],
                    {},
                )
        hooks = {"trace": trace, "runtime": runtime, "kind": self.kind}
        await invoke(self.task.validation_before, hooks)
        if self.config.command:
            result = await solver.run(self.config.command, {})
            trace.info["oracle"] = {
                "argv": self.config.command,
                "user": solver.user,
                "exit_code": result.exit_code,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        await invoke(self.task.validation_after, hooks)
        if self.config.source is not None:
            await self.adopt_reply(trace, runtime)
        # A broken reference is still graded: its failure and the actual grade
        # are both useful evidence, just as for an ordinary solver's attempt.
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def adopt_reply(self, trace: Trace, runtime: Runtime) -> None:
        """Make the gold solution's `GOLD_REPLY`, when it left one, the trace's final
        assistant message, after the task's prompt. Neither message was sampled."""
        path = str(PurePosixPath(self.config.target_dir, GOLD_REPLY))
        probe = await runtime.run(["test", "-f", path], {})
        if probe.exit_code:
            return
        reply = (await runtime.read(path, max_bytes=MAX_GOLD_REPLY_BYTES)).decode(
            "utf-8", errors="replace"
        )
        parent = len(trace.nodes) - 1 if trace.nodes else None
        if parent is None:
            trace.nodes.append(
                MessageNode(message=UserMessage(content=self.task.data.prompt_text))
            )
            parent = 0
        trace.nodes.append(
            MessageNode(parent=parent, message=AssistantMessage(content=reply))
        )
        trace._head_index = {}


async def validate_attempt(
    task: Task,
    runtime: Runtime | RuntimeConfig,
    *,
    source: Path | None = None,
    target_dir: str = "/solution",
    command: list[str] | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_rewards: dict[str, float] | None = None,
    maximum_rewards: dict[str, float] | None = None,
    timeout: TimeoutConfig | None = None,
    retries: RetryConfig | None = None,
    on_trace: Callable[[Trace], None] | None = None,
    run_attempt: ValidationRunner | None = None,
) -> ValidationAttempt:
    """Execute one reference or untouched attempt in its own native deployment."""
    from verifiers.v1.agent import Agent

    class ValidationAgent(Agent):
        async def _run_once(self, task, *args, **kwargs):
            # Routing may replace the task. Hooks must see the same controller
            # whose setup/finalize/scoring are run by the native rollout.
            self.harness.task = task
            return await super()._run_once(task, *args, **kwargs)

    task = copy.deepcopy(task)
    config = runtime.config if isinstance(runtime, Runtime) else runtime
    if isinstance(config, PrimeConfig) and config.checkpoint:
        config = config.model_copy(update={"checkpoint": None})
    harness_config = ValidationHarnessConfig(
        source=source, target_dir=target_dir, command=command or []
    )
    agent_config = AgentConfig(
        runtime=config,
        model="verifiers/validation",
        timeout=timeout or TimeoutConfig(),
        retries=retries or RetryConfig(),
    ).model_copy(update={"harness": harness_config})
    harness = ValidationHarness(harness_config)
    harness.task = task
    harness.kind = "gold" if source is not None else "baseline"
    agent = ValidationAgent(agent_config)
    agent.harness = harness
    agent.trainable = False

    def started(trace: Trace) -> None:
        trace.agent.trainable = False
        trace.agent.name = harness.kind
        if on_trace is not None:
            on_trace(trace)

    try:
        if run_attempt is not None:
            trace = await run_attempt(agent, task, started)
        else:
            trace = await agent.run(
                task,
                on_score=getattr(task, "judge_on_score", None),
                on_trace=started,
                durable=False,
            )
    except Exception as error:  # noqa: BLE001 - report preflight failures as evidence
        infrastructure = not isinstance(error, TaskError)
        return ValidationAttempt(
            valid=None if infrastructure else False,
            error=f"{type(error).__name__}: {error}",
            infrastructure=infrastructure,
        )
    incomplete = not trace.ok or "timeout" in (trace.stop_condition or "")
    error = trace.last_error if not trace.ok else None
    infrastructure = incomplete and (error is None or error.type != "TaskError")
    reward = trace.reward if not incomplete and trace.rewards else None
    valid = (
        (minimum is None or reward >= minimum)
        and (maximum is None or reward <= maximum)
        if reward is not None
        else False
        if incomplete and not infrastructure
        else None
    )
    if not incomplete and (minimum_rewards or maximum_rewards):
        valid = all(
            name in trace.rewards
            and trace.rewards[name] is not None
            and trace.rewards[name].score >= bound
            for name, bound in (minimum_rewards or {}).items()
        ) and all(
            name in trace.rewards
            and trace.rewards[name] is not None
            and trace.rewards[name].score <= bound
            for name, bound in (maximum_rewards or {}).items()
        )
    oracle = trace.info.get("oracle") or {}
    command_error = (
        f"Reference command exited with code {oracle['exit_code']}"
        if not incomplete and oracle.get("exit_code", 0) != 0
        else None
    )
    if command_error is not None:
        valid = False
    return ValidationAttempt(
        reward=reward,
        valid=valid,
        trace=trace.model_dump(mode="json"),
        error=(
            f"{error.type}: {error.message}"
            if error is not None
            else trace.stop_condition
            if incomplete
            else command_error
        ),
        infrastructure=infrastructure,
    )


async def validate_pair(
    task: Task,
    runtime: Runtime | RuntimeConfig,
    *,
    baseline: bool = True,
    gold: bool = True,
    baseline_max: float = 0.0,
    gold_min: float = 1.0,
    gold_rewards: dict[str, float] | None = None,
    baseline_rewards: list[str] | None = None,
    gold_source: Path | None = None,
    gold_target_dir: str = "/solution",
    gold_command: list[str] | None = None,
    timeout: TimeoutConfig | None = None,
    retries: RetryConfig | None = None,
    on_trace: Callable[[Trace], None] | None = None,
    run_attempt: ValidationRunner | None = None,
) -> ValidationResult:
    """Grade untouched and gold work in independent native deployments.

    The gold attempt places `gold_source` at `gold_target_dir` and runs
    `gold_command` (default: its solve.sh). Each reward must meet its inclusive
    bound. Callers may repeat this check concurrently; neither attempt reuses
    the caller's live workspace.
    """
    runs = {}
    results = {
        "baseline": ValidationAttempt(enabled=False),
        "gold": ValidationAttempt(enabled=False),
    }
    if baseline:
        runs["baseline"] = validate_attempt(
            task,
            runtime,
            maximum=None if baseline_rewards else baseline_max,
            maximum_rewards={name: baseline_max for name in baseline_rewards}
            if baseline_rewards
            else None,
            timeout=timeout,
            retries=retries,
            on_trace=on_trace,
            run_attempt=run_attempt,
        )
    if gold and (
        gold_source is None
        or not gold_source.is_dir()
        or (gold_command is None and not (gold_source / "solve.sh").is_file())
    ):
        results["gold"] = ValidationAttempt(
            error=f"gold solution unavailable: {gold_source}"
        )
    elif gold:
        runs["gold"] = validate_attempt(
            task,
            runtime,
            source=gold_source,
            target_dir=gold_target_dir,
            command=(
                gold_command
                if gold_command is not None
                else ["bash", f"{gold_target_dir.rstrip('/')}/solve.sh"]
            ),
            minimum=None if gold_rewards else gold_min,
            minimum_rewards=gold_rewards,
            timeout=timeout,
            retries=retries,
            on_trace=on_trace,
            run_attempt=run_attempt,
        )
    results.update(zip(runs, await asyncio.gather(*runs.values())))
    return ValidationResult(**results)
