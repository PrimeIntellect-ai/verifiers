"""Model-free validation through the ordinary rollout and scoring lifecycle."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from verifiers.v1.configs.agent import AgentConfig, TimeoutConfig
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.configs.retries import RetryConfig
from verifiers.v1.errors import TaskError
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import PrimeConfig, ProgramResult, Runtime, RuntimeConfig
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

__all__ = ["ValidationAttempt", "ValidationHarness", "ValidationResult"]


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

        runtime = runtime.with_user(None)
        if self.config.source is not None:
            for source in sorted(self.config.source.rglob("*")):
                if source.is_file():
                    relative = source.relative_to(self.config.source).as_posix()
                    await runtime.write(
                        str(PurePosixPath(self.config.target_dir, relative)),
                        source.read_bytes(),
                    )
        hooks = {"trace": trace, "runtime": runtime, "kind": self.kind}
        await invoke(self.task.validation_before, hooks)
        if self.config.command:
            result = await runtime.run(self.config.command, {})
            trace.info["oracle"] = {
                "argv": self.config.command,
                "exit_code": result.exit_code,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        await invoke(self.task.validation_after, hooks)
        # A broken reference is still graded: its failure and the actual grade
        # are both useful evidence, just as for an ordinary solver's attempt.
        return ProgramResult(exit_code=0, stdout="", stderr="")


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
) -> ValidationAttempt:
    """Execute one reference or untouched attempt in its own native deployment."""
    from verifiers.v1.agent import Agent

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
    agent = Agent(agent_config)
    agent.harness = harness
    agent.trainable = False

    def started(trace: Trace) -> None:
        trace.agent.trainable = False
        trace.agent.name = harness.kind
        if on_trace is not None:
            on_trace(trace)

    try:
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
    error = trace.last_error
    incomplete = not trace.ok or "timeout" in (trace.stop_condition or "")
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
    return ValidationAttempt(
        reward=reward,
        valid=valid,
        trace=trace.model_dump(mode="json"),
        error=(
            f"{error.type}: {error.message}"
            if error is not None
            else trace.stop_condition
            if incomplete
            else None
        ),
        infrastructure=infrastructure,
    )
